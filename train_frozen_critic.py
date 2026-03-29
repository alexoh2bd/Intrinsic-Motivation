"""
Experiment 3: Full-Depth Actor from Scratch with Frozen Critic

Freeze a pre-trained critic (InfoNCE or SIGReg) and train a fresh deep
residual Actor (same architecture as the original) from scratch.  The
critic provides the Q-landscape; only the actor + entropy coefficient
are updated.

Usage:
    uv run train_shallow_actor.py --checkpoint_path runs/humanoid_infonce
    uv run train_shallow_actor.py --checkpoint_path runs/humanoid_infonce --override_env_id humanoid_U_maze
"""
import os
import jax
import flax
import tyro
import time
import optax
import wandb
import pickle
import random
import wandb_osh
import numpy as np
import flax.linen as nn
import jax.numpy as jnp

from brax import envs
from etils import epath
from dataclasses import dataclass
from typing import NamedTuple, Any
from wandb_osh.hooks import TriggerWandbSyncHook
from flax.training.train_state import TrainState
from flax.linen.initializers import variance_scaling
from brax.io import html
from flax.core import freeze, unfreeze

# ── Import shared definitions from train.py (which re-exports from src/) ────
from train import (
    lecun_unfirom,
    bias_init,
    residual_block,
    UnifiedEncoder,
    SA_encoder,
    G_encoder,
    Actor,
    TrainingState,
    Transition,
    load_params,
    save_params,
)
from train import _make_env  # shared env factory
from src.loss import SIGRegModule, lejepa_loss, sigreg_forward, tri_loss
from src.evaluator import CrlEvaluator
from src.buffer import TrajectoryUniformSamplingQueue


# ─── Args ────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    exp_name: str = "frozen_critic_actor"
    seed: int = 1000
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "JaxGCRL_testing"
    wandb_entity: str = 'aho13-duke-university'
    wandb_mode: str = 'offline'
    wandb_dir: str = '.'
    wandb_group: str = '.'
    capture_vis: bool = True
    vis_length: int = 1000
    checkpoint: bool = True

    # ── Frozen-critic experiment specific ──
    checkpoint_path: str = ""
    """Path to a runs/ directory containing args.pkl and final.pkl (or a specific step .pkl)"""
    checkpoint_file: str = "final.pkl"
    """Which checkpoint .pkl to load from checkpoint_path (default: final.pkl)"""

    # ── Actor config (full-depth residual actor, matching critic architecture) ──
    actor_network_width: int = 256
    """Hidden-layer width for the fresh actor (default matches critic width)"""
    actor_depth: int = 64
    """Number of layers for the fresh actor (default matches critic depth)"""
    actor_skip_connections: int = 4
    """Skip connection frequency for the actor (0 = use residual blocks from Actor class)"""

    # ── Environment (will be overridden from saved args.pkl unless override_env_id set) ──
    override_env_id: str = ""
    """If set, use this env for training instead of the checkpoint's env_id (e.g. humanoid_U_maze)"""
    env_id: str = "humanoid"
    episode_length: int = 1000
    obs_dim: int = 0
    goal_start_idx: int = 0
    goal_end_idx: int = 0

    # ── Training ──
    total_env_steps: int = 100000000
    num_epochs: int = 100
    num_envs: int = 512
    eval_env_id: str = ""
    num_eval_envs: int = 128
    actor_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 62

    # Critic architecture (loaded from checkpoint, used for reconstruction)
    critic_network_width: int = 256
    critic_depth: int = 4
    critic_skip_connections: int = 0
    use_relu: int = 0

    num_episodes_per_env: int = 1
    training_steps_multiplier: int = 1
    use_all_batches: int = 0
    num_sgd_batches_per_training_step: int = 800

    eval_actor: int = 0
    expl_actor: int = 1

    entropy_param: float = 0.5
    disable_entropy: int = 0
    num_render: int = 10
    save_buffer: int = 0

    # computed at runtime
    env_steps_per_actor_step: int = 0
    num_prefill_env_steps: int = 0
    num_prefill_actor_steps: int = 0
    num_training_steps_per_epoch: int = 0

    sigreg: bool = False
    lamb: float = 0.05
    logsumexp_penalty_coeff: float = 0.1

if __name__ == "__main__":

    args = tyro.cli(Args)

    # ── Load saved args from the checkpoint to recover env / critic config ──
    assert args.checkpoint_path, "Must provide --checkpoint_path (e.g. runs/humanoid_infonce)"
    saved_args_path = os.path.join(args.checkpoint_path, "args.pkl")
    with open(saved_args_path, 'rb') as f:
        saved_args = pickle.load(f)

    # Inherit env & critic config from the saved run
    for attr in [
        'env_id', 'episode_length', 'obs_dim', 'goal_start_idx', 'goal_end_idx',
        'critic_network_width', 'critic_depth', 'critic_skip_connections',
        'use_relu', 'sigreg', 'lamb', 'logsumexp_penalty_coeff',
    ]:
        setattr(args, attr, getattr(saved_args, attr))

    # Default actor dimensions to match the checkpoint's critic unless overridden
    # (CLI values != dataclass defaults means user explicitly set them)
    args.actor_network_width = args.critic_network_width
    args.actor_depth = args.critic_depth

    # Allow overriding the training env (e.g. train on U-maze with humanoid-pretrained critic)
    if args.override_env_id:
        print(f"Overriding env_id: {args.env_id} -> {args.override_env_id}", flush=True)
        args.env_id = args.override_env_id

    print("=" * 60, flush=True)
    print("EXPERIMENT 3: Full-Depth Actor from Scratch + Frozen Critic", flush=True)
    print("=" * 60, flush=True)
    print(f"Checkpoint: {args.checkpoint_path}/{args.checkpoint_file}", flush=True)
    print(f"Critic trained on: {getattr(saved_args, 'env_id', 'unknown')}", flush=True)
    print(f"Training env: {args.env_id}", flush=True)
    print(f"Critic type: {'SIGReg + InfoNCE' if args.sigreg else 'InfoNCE'}", flush=True)
    print(f"Actor: depth={args.actor_depth}, width={args.actor_network_width} (residual)", flush=True)
    print(f"Critic: depth={args.critic_depth}, width={args.critic_network_width}", flush=True)
    print(flush=True)

    # Print all args
    print("Arguments:", flush=True)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}", flush=True)
    print(flush=True)

    # ── Derived quantities ──
    args.env_steps_per_actor_step = args.num_envs * args.unroll_length
    args.num_prefill_env_steps = args.min_replay_size * args.num_envs
    args.num_prefill_actor_steps = np.ceil(args.min_replay_size / args.unroll_length)
    args.num_training_steps_per_epoch = (
        (args.total_env_steps - args.num_prefill_env_steps)
        // (args.num_epochs * args.env_steps_per_actor_step)
    )

    sigma = "sigreg" if args.sigreg else "infonce"
    run_name = (
        f"frozen_critic_{args.env_id}_actor{args.actor_depth}x{args.actor_network_width}"
        f"_critic{args.critic_depth}x{args.critic_network_width}_{sigma}_{args.seed}"
    )

    if args.track:
        if args.wandb_group == '.':
            args.wandb_group = None

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            group=args.wandb_group,
            dir=args.wandb_dir,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

        if args.wandb_mode == 'offline':
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()

    if args.checkpoint:
        from pathlib import Path
        from datetime import datetime
        short_run_name = f"runs/frozen_critic_{args.env_id}_{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        save_path = Path(args.wandb_dir) / Path(short_run_name)
        os.mkdir(path=save_path)

    # ── RNG ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, buffer_key, env_key, eval_env_key, actor_key, sa_key, g_key = jax.random.split(key, 7)

    # ── Environment setup (uses shared factory from train.py -> src/) ──
    def make_env(env_id=args.env_id):
        return _make_env(env_id, args)

    env = make_env()
    env = envs.training.wrap(env, episode_length=args.episode_length)

    obs_size = env.observation_size
    action_size = env.action_size
    env_keys = jax.random.split(env_key, args.num_envs)
    env_state = jax.jit(env.reset)(env_keys)
    env.step = jax.jit(env.step)

    print(f"obs_size: {obs_size}, action_size: {action_size}", flush=True)

    if not args.eval_env_id:
        args.eval_env_id = args.env_id

    eval_env = make_env(args.eval_env_id)
    eval_env = envs.training.wrap(eval_env, episode_length=args.episode_length)
    eval_env_keys = jax.random.split(eval_env_key, args.num_envs)
    eval_env_state = jax.jit(eval_env.reset)(eval_env_keys)
    eval_env.step = jax.jit(eval_env.step)

    # ── Load checkpoint ──
    ckpt_file = os.path.join(args.checkpoint_path, args.checkpoint_file)
    print(f"Loading checkpoint from: {ckpt_file}", flush=True)
    alpha_params_loaded, actor_params_loaded, critic_params_loaded = load_params(ckpt_file)

    # Auto-detect sigreg vs infonce from checkpoint structure
    is_sigreg = "u_encoder" in critic_params_loaded
    if is_sigreg != args.sigreg:
        print(f"WARNING: Overriding sigreg={args.sigreg} -> {is_sigreg} based on checkpoint structure", flush=True)
        args.sigreg = is_sigreg

    print(f"Critic params keys: {list(critic_params_loaded.keys())}", flush=True)
    print(f"Detected critic type: {'SIGReg (UnifiedEncoder)' if args.sigreg else 'InfoNCE (SA+G encoders)'}", flush=True)

    # ── Reconstruct critic modules (for forward pass only, params are frozen) ──
    if args.sigreg:
        global sigreg_params
        sigreg_params = SIGRegModule.init_sigreg_params(knots=17)

        u_encoder = UnifiedEncoder(
            network_width=args.critic_network_width,
            network_depth=args.critic_depth,
            skip_connections=args.critic_skip_connections,
            use_relu=args.use_relu,
        )
    else:
        sa_encoder = SA_encoder(
            network_width=args.critic_network_width,
            network_depth=args.critic_depth,
            skip_connections=args.critic_skip_connections,
            use_relu=args.use_relu,
        )
        g_encoder = G_encoder(
            network_width=args.critic_network_width,
            network_depth=args.critic_depth,
            skip_connections=args.critic_skip_connections,
            use_relu=args.use_relu,
        )

    # SIGReg params for the actor JEPA loss (separate from critic's sigreg_params)
    sigreg_params_actor = SIGRegModule.init_sigreg_params(knots=17)

    # Frozen critic state — use optax.set_to_zero() so no gradient updates happen
    critic_state = TrainState.create(
        apply_fn=None,
        params=critic_params_loaded,
        tx=optax.set_to_zero(),   # <-- frozen: zero optimizer = no param updates
    )

    # ── Full-depth residual Actor (fresh random init) ──
    actor = Actor(
        action_size=action_size,
        network_width=args.actor_network_width,
        network_depth=args.actor_depth,
        skip_connections=args.actor_skip_connections,
        use_relu=args.use_relu,
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, np.ones([1, obs_size])),
        tx=optax.adam(learning_rate=args.actor_lr),
    )

    print(f"Fresh actor initialised: depth={args.actor_depth}, width={args.actor_network_width}, "
          f"residual blocks={args.actor_depth // 4}", flush=True)

    # Entropy coefficient (fresh)
    target_entropy = -args.entropy_param * action_size
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": log_alpha},
        tx=optax.adam(learning_rate=args.alpha_lr),
    )

    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
    )

    # ── Replay Buffer ──
    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))

    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        extras={
            "state_extras": {
                "truncation": 0.0,
                "seed": 0.0,
            }
        },
    )

    def jit_wrap(buffer):
        buffer.insert_internal = jax.jit(buffer.insert_internal)
        buffer.sample_internal = jax.jit(buffer.sample_internal)
        return buffer

    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=args.max_replay_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=args.batch_size,
            num_envs=args.num_envs,
            episode_length=args.episode_length,
        )
    )
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    # ── Actor / eval step functions ──

    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        means, _, _ = actor.apply(training_state.actor_state.params, env_state.obs)
        actions = nn.tanh(means)
        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1-nstate.done,
            extras={"state_extras": state_extras},
        )

    def actor_step(training_state, env, env_state, key, extra_fields):
        means, log_stds, _ = actor.apply(training_state.actor_state.params, env_state.obs)
        stds = jnp.exp(log_stds)
        actions = nn.tanh(means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype))
        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1-nstate.done,
            extras={"state_extras": state_extras},
        )

    extra_fields = ("truncation", "seed")

    @jax.jit
    def get_experience(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            env_state, transition = actor_step(training_state, env, env_state, current_key, extra_fields)
            return (env_state, next_key), transition

        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=args.unroll_length)

        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        def f(carry, unused):
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state = get_experience(
                training_state, env_state, buffer_state, key
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + args.env_steps_per_actor_step
            )
            return (training_state, env_state, buffer_state, new_key), ()

        (training_state, env_state, buffer_state, _), _ = jax.lax.scan(
            f, (training_state, env_state, buffer_state, key), (),
            length=args.num_prefill_actor_steps
        )
        return training_state, env_state, buffer_state, key

    # ── Update functions ──

    @jax.jit
    def update_actor_and_alpha(transitions, training_state, key):
        """Train ONLY the actor (critic is frozen)."""
        actor_batch_size = args.batch_size
        transitions = jax.tree_util.tree_map(
            lambda x: x[:actor_batch_size],
            transitions,
        )

        def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
            obs = transitions.observation
            state = obs[:, :args.obs_dim]
            future_state = transitions.extras["future_state"]
            goal = future_state[:, args.goal_start_idx : args.goal_end_idx]
            observation = jnp.concatenate([state, goal], axis=1)

            key, sample_key, sig_key1, sig_key2 = jax.random.split(key, 4)

            means, log_stds, _ = actor.apply(actor_params, observation)
            stds = jnp.exp(log_stds)
            x_ts = means + stds * jax.random.normal(sample_key, shape=means.shape, dtype=means.dtype)
            action = nn.tanh(x_ts)
            log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
            log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
            log_prob = log_prob.sum(-1)

            # Forward through FROZEN critic (stop_gradient is belt-and-suspenders
            # since the optimizer is set_to_zero, but makes intent clear)
            frozen_critic = jax.lax.stop_gradient(critic_params)

            if args.sigreg:
                sa_repr = u_encoder.apply(
                    frozen_critic["u_encoder"],
                    jnp.concatenate([state, action], axis=-1),
                    input_type="state_action",
                )
                g_repr = u_encoder.apply(
                    frozen_critic["u_encoder"],
                    goal,
                    input_type="goal",
                )
            else:
                sa_repr = sa_encoder.apply(frozen_critic["sa_encoder"], state, action)
                g_repr = g_encoder.apply(frozen_critic["g_encoder"], goal)

            qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))

            # Actor loss: maximise Q (minimise distance)
            actor_loss = -jnp.mean(qf_pi)

            # JEPA-style SIGReg regularisation on both representation streams
            # (only when critic was trained with SIGReg — keeps InfoNCE baseline clean)
            if args.sigreg:
                actor_loss = actor_loss + (
                    args.lamb * sigreg_forward(sa_repr, sigreg_params_actor, sig_key1)
                    + args.lamb * sigreg_forward(g_repr,  sigreg_params_actor, sig_key2)
                )

            if not args.disable_entropy:
                actor_loss = actor_loss + jnp.mean(jnp.exp(log_alpha) * log_prob)

            return actor_loss, log_prob

        def alpha_loss(alpha_params, log_prob):
            alpha = jnp.exp(alpha_params["log_alpha"])
            alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - target_entropy))
            return jnp.mean(alpha_loss)

        (actorloss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
            training_state.actor_state.params,
            training_state.critic_state.params,
            training_state.alpha_state.params['log_alpha'],
            transitions,
            key,
        )
        new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

        alphaloss, alpha_grad = jax.value_and_grad(alpha_loss)(
            training_state.alpha_state.params, log_prob
        )
        new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

        training_state = training_state.replace(
            actor_state=new_actor_state, alpha_state=new_alpha_state
        )

        metrics = {
            "sample_entropy": -log_prob,
            "actor_loss": actorloss,
            "alpha_loss": alphaloss,
            "log_alpha": training_state.alpha_state.params["log_alpha"],
        }

        return training_state, metrics

    # ── SGD step: actor only (NO critic update) ──

    @jax.jit
    def sgd_step(carry, transitions):
        training_state, key = carry
        key, actor_key = jax.random.split(key)

        training_state, actor_metrics = update_actor_and_alpha(
            transitions, training_state, actor_key
        )

        training_state = training_state.replace(
            gradient_steps=training_state.gradient_steps + 1
        )

        return (training_state, key), actor_metrics

    # ── Training step & epoch (same structure as train.py) ──

    @jax.jit
    def training_step(training_state, env_state, buffer_state, key, t):
        experience_key1, experience_key2, sampling_key, training_key, sgd_batches_key = jax.random.split(key, 5)

        env_state, buffer_state = get_experience(
            training_state, env_state, buffer_state, experience_key1,
        )

        training_state = training_state.replace(
            env_steps=training_state.env_steps + args.env_steps_per_actor_step,
        )

        transitions_list = []
        for _ in range(args.num_episodes_per_env):
            buffer_state, new_transitions = replay_buffer.sample(buffer_state)
            transitions_list.append(new_transitions)

        transitions = jax.tree_util.tree_map(
            lambda *arrays: jnp.concatenate(arrays, axis=0),
            *transitions_list,
        )

        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(
            TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0)
        )(
            (args.gamma, args.obs_dim, args.goal_start_idx, args.goal_end_idx),
            transitions,
            batch_keys,
        )

        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )

        permutation = jax.random.permutation(experience_key2, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)

        num_full_batches = len(transitions.observation) // args.batch_size
        transitions = jax.tree_util.tree_map(
            lambda x: x[:num_full_batches * args.batch_size], transitions
        )

        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]),
            transitions,
        )

        if args.use_all_batches == 0:
            num_total_batches = transitions.observation.shape[0]
            selected_indices = jax.random.permutation(
                sgd_batches_key, num_total_batches
            )[:args.num_sgd_batches_per_training_step]
            transitions = jax.tree_util.tree_map(
                lambda x: x[selected_indices], transitions
            )

        (training_state, _), metrics = jax.lax.scan(
            sgd_step, (training_state, training_key), transitions
        )

        return (training_state, env_state, buffer_state), metrics

    @jax.jit
    def training_epoch(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, t):
            ts, es, bs, k = carry
            k, train_key = jax.random.split(k, 2)
            (ts, es, bs), metrics = training_step(ts, es, bs, train_key, t)
            return (ts, es, bs, k), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            jnp.arange(args.num_training_steps_per_epoch * args.training_steps_multiplier),
        )

        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

    # ── Prefill ──
    key, prefill_key = jax.random.split(key, 2)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key
    )

    # ── Evaluator ──
    if args.eval_actor == 0:
        from functools import partial
        evaluator = CrlEvaluator(
            partial(deterministic_actor_step, extra_fields=extra_fields),
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )
    else:
        from functools import partial
        evaluator = CrlEvaluator(
            partial(deterministic_actor_step, extra_fields=extra_fields),
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )

    # ── Training loop ──
    training_walltime = 0
    print('Starting frozen-critic actor training...', flush=True)
    start_time = time.time()

    for ne in range(args.num_epochs):
        t = time.time()

        key, epoch_key = jax.random.split(key)
        training_state, env_state, buffer_state, metrics = training_epoch(
            training_state, env_state, buffer_state, epoch_key
        )

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time

        sps = (args.env_steps_per_actor_step * args.num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            "training/envsteps": training_state.env_steps.item(),
            **{f"training/{name}": value for name, value in metrics.items()},
        }

        metrics = evaluator.run_evaluation(training_state, metrics)

        print(f"epoch {ne}/{args.num_epochs} | metrics: {metrics}", flush=True)

        if args.checkpoint:
            if ne < 5 or ne >= args.num_epochs - 5 or ne % 10 == 0:
                params = (
                    training_state.alpha_state.params,
                    training_state.actor_state.params,
                    training_state.critic_state.params,
                )
                path = f"{save_path}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)
                jax.clear_caches()

        if args.track:
            wandb.log(metrics, step=ne)
            if args.wandb_mode == 'offline':
                trigger_sync()

        hours_passed = (time.time() - start_time) / 3600
        print(f"Time elapsed: {hours_passed:.3f} hours", flush=True)

    # ── Final save ──
    if args.checkpoint:
        params = (
            training_state.alpha_state.params,
            training_state.actor_state.params,
            training_state.critic_state.params,
        )
        save_params(f"{save_path}/final.pkl", params)

    # ── Render ──
    if args.capture_vis:
        def render_policy(training_state, save_path):
            @jax.jit
            def policy_step(env_state, actor_params):
                means, _, _ = actor.apply(actor_params, env_state.obs)
                actions = nn.tanh(means)
                next_state = env.step(env_state, actions)
                return next_state, env_state

            rollout_states = []
            for i in range(args.num_render):
                render_env = make_env(args.eval_env_id)
                rng = jax.random.PRNGKey(seed=i+1)
                env_st = jax.jit(render_env.reset)(rng)
                for _ in range(args.vis_length):
                    env_st, current_state = policy_step(env_st, training_state.actor_state.params)
                    rollout_states.append(current_state.pipeline_state)

            html_string = html.render(render_env.sys, rollout_states)
            render_path = f"{save_path}/vis.html"
            with open(render_path, "w") as f_out:
                f_out.write(html_string)
            wandb.log({"vis": wandb.Html(html_string)})

        print("Rendering final policy...", flush=True)
        try:
            render_policy(training_state, save_path)
        except Exception as e:
            print(f"Error rendering final policy: {e}", flush=True)

    if args.checkpoint:
        with open(f"{save_path}/args.pkl", 'wb') as f:
            pickle.dump(args, f)
        print(f"Saved args to {save_path}/args.pkl", flush=True)

    print("Done!", flush=True)
