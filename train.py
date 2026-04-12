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

# ── shared src/ imports ──────────────────────────────────────────────────────
from src.networks import (
    lecun_unfirom,
    bias_init,
    residual_block,
    UnifiedEncoder,
    SA_encoder,
    G_encoder,
    Actor,
)
from src.types import TrainingState, Transition
from src.utils import load_params, save_params
from src.env_factory import make_env as _make_env
from src.loss import SIGRegModule, lejepa_loss, sigreg_forward, sigreg_iso, tri_loss
from src.evaluator import CrlEvaluator
from src.buffer import TrajectoryUniformSamplingQueue


@dataclass
class Args:
    exp_name: str = "train"
    seed: int = 1000
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "JaxGCRL_testing"
    wandb_entity: str = 'aho13-duke-university'
    wandb_mode: str = 'online'
    wandb_dir: str = '.'
    wandb_group: str = '.'
    capture_vis: bool = True
    vis_length: int = 1000
    checkpoint: bool = True

    #environment specific arguments
    env_id: str = "humanoid" # "ant_big_maze" "humanoid_u_maze" "arm_binpick_hard"
    episode_length: int = 1000
    # to be filled in runtime
    obs_dim: int = 0
    goal_start_idx: int = 0
    goal_end_idx: int = 0

    # Algorithm specific arguments
    total_env_steps: int = 100000000 # 50000000
    num_epochs: int = 100 # 50
    num_envs: int = 512
    eval_env_id: str = ""
    num_eval_envs: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    logsumexp_penalty_coeff: float = 0.1

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    
    unroll_length: int  = 62

    critic_network_width: int = 256
    actor_network_width: int = 256
    actor_depth: int = 4
    critic_depth: int = 4
    actor_skip_connections: int = 0 # 0 for no skip connections, >= 0 means the frequency of skip connections (every N layers)
    critic_skip_connections: int = 0 # 0 for no skip connections, >= 0 means the frequency of skip connections (every N layers)
    
    num_episodes_per_env: int = 1 #recommended to keep at 1
    training_steps_multiplier: int = 1 #recommended to keep at 1
    use_all_batches: int = 0 # recommended to keep at 0
    num_sgd_batches_per_training_step: int = 800
    
    eval_actor: int = 0 # recommended to keep at 0
    # if 0, use deterministic actor for evaluation
    # if 1, use stochastic actor for evaluation
    # if 2, sample two actions and take the one with the higher Q value
    # if K >= 2, sample K actions and take the one with the highest Q value
    expl_actor: int = 1 # recommended to keep at 1
    # if 0, use deterministic actor for exploration/collecting data
    # if 1, use stochastic actor for exploration/collecting data
    # if 2, sample two actions and take the one with the higher Q value
    # if K >= 2, sample K actions and take the one with the highest Q value
    
    entropy_param: float = 0.5
    disable_entropy: int = 0
    use_relu: int = 0
    num_render: int = 10
    save_buffer: int = 0
    
    # to be filled in runtime
    env_steps_per_actor_step : int = 0
    """number of env steps per actor step (computed in runtime)"""
    num_prefill_env_steps : int = 0
    """number of env steps to fill the buffer before starting training (computed in runtime)"""
    num_prefill_actor_steps : int = 0
    """number of actor steps to fill the buffer before starting training (computed in runtime)"""
    num_training_steps_per_epoch : int = 0
    """the number of training steps per epoch(computed in runtime)"""

    unified_encoder: bool = False  # Use UnifiedEncoder for critic (shared trunk, two input projections)
    sigreg: bool = False            # Use SIGReg/JEPA loss (auto-enables unified_encoder)
    lamb: float = 0.05

    # Actor embedding regularization (SIGReg on backbone output before action heads)
    actor_embed_reg: bool = False
    actor_reg_coeff: float = 0.05
    # ISO actor: sigreg_iso (isotropic Gaussian) instead of sigreg_forward; parameter-free
    iso_actor: bool = False
    sigreg_iso_num_slices: int = 16
    sigreg_iso_num_t: int = 8
    sigreg_iso_t_max: float = 5.0


# Network classes (UnifiedEncoder, SA_encoder, G_encoder, Actor),
# TrainingState, Transition, load_params, save_params, residual_block,
# lecun_unfirom, bias_init  — all imported from src/ above.

if __name__ == "__main__":

    args = tyro.cli(Args)
    
    # Print every arg
    print("Arguments:", flush=True)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}", flush=True)
    print("\n", flush=True)

    args.env_steps_per_actor_step = args.num_envs * args.unroll_length
    print(f"env_steps_per_actor_step: {args.env_steps_per_actor_step}", flush=True)

    args.num_prefill_env_steps = args.min_replay_size * args.num_envs
    print(f"num_prefill_env_steps: {args.num_prefill_env_steps}", flush=True)

    args.num_prefill_actor_steps = np.ceil(args.min_replay_size / args.unroll_length)
    print(f"num_prefill_actor_steps: {args.num_prefill_actor_steps}", flush=True)

    args.num_training_steps_per_epoch = (args.total_env_steps - args.num_prefill_env_steps) // (args.num_epochs * args.env_steps_per_actor_step)
    print(f"num_training_steps_per_epoch: {args.num_training_steps_per_epoch}", flush=True)

    # SIGReg loss requires the unified encoder; auto-enable for backward compat
    if args.sigreg and not args.unified_encoder:
        print("NOTE: --sigreg requires --unified_encoder; enabling unified_encoder automatically.", flush=True)
        args.unified_encoder = True

    encoder_tag = "unified" if args.unified_encoder else "separate"
    loss_tag = "sigreg" if args.sigreg else "infonce"
    sigma = f"{encoder_tag}_{loss_tag}"
    run_name = f"{args.env_id}{'_' + args.eval_env_id if args.eval_env_id else ''}_{args.batch_size}_{args.total_env_steps}_nenvs:{args.num_envs}_criticwidth:{args.critic_network_width}_actorwidth:{args.actor_network_width}_criticdepth:{args.critic_depth}_actordepth:{args.actor_depth}_actorskip:{args.actor_skip_connections}_criticskip:{args.critic_skip_connections}_{args.seed}_{sigma}"
    print(f"run_name: {run_name}", flush=True)
    
    if args.track:

        if args.wandb_group ==  '.':
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
        short_run_name = f"runs/{args.env_id}_{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        save_path = Path(args.wandb_dir) / Path(short_run_name)
        os.mkdir(path=save_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, buffer_key, env_key, eval_env_key, actor_key, sa_key, g_key = jax.random.split(key, 7)

    # Use shared env factory from src/
    def make_env(env_id=args.env_id):
        return _make_env(env_id, args)

    env = make_env()
    env = envs.training.wrap(
        env,
        episode_length=args.episode_length,
    )

    obs_size = env.observation_size
    action_size = env.action_size
    env_keys = jax.random.split(env_key, args.num_envs)
    env_state = jax.jit(env.reset)(env_keys)
    env.step = jax.jit(env.step)
    
    print(f"obs_size: {obs_size}, action_size: {action_size}", flush=True)
    
    
    if not args.eval_env_id:
        args.eval_env_id = args.env_id
        
    # make eval env
    eval_env = make_env(args.eval_env_id)
    eval_env = envs.training.wrap(
        eval_env,
        episode_length=args.episode_length,
    )
    eval_env_keys = jax.random.split(eval_env_key, args.num_envs)
    eval_env_state = jax.jit(eval_env.reset)(eval_env_keys)
    eval_env.step = jax.jit(eval_env.step)

        
    # Network setup
    # Actor
    actor = Actor(action_size=action_size, network_width=args.actor_network_width, network_depth=args.actor_depth, skip_connections=args.actor_skip_connections, use_relu=args.use_relu)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, np.ones([1, obs_size])),
        tx=optax.adam(learning_rate=args.actor_lr)
    )

    # Actor embedding SIGReg params (separate from critic's sigreg_params)
    # iso_actor uses sigreg_iso which is parameter-free, so no params needed
    if args.actor_embed_reg and not args.iso_actor:
        sigreg_params_actor = SIGRegModule.init_sigreg_params(knots=17)

    # Critic — architecture choice (unified vs. separate encoders)
    if args.unified_encoder:
        # Unified encoder for both goals and state-actions
        u_encoder = UnifiedEncoder(
            network_width=args.critic_network_width, 
            network_depth=args.critic_depth, 
            skip_connections=args.critic_skip_connections, 
            use_relu=args.use_relu
        )
        # Initialize BOTH projection paths by tracing each input_type separately
        # This ensures both goal_projection and sa_projection layers are created
        sa_key, g_init_key = jax.random.split(sa_key)
        
        # First init: trace state-action path
        sa_params = u_encoder.init(
            sa_key, 
            np.ones([1, args.obs_dim + action_size]),
            input_type="state_action"
        )
        
        # Second init: trace goal path
        g_params = u_encoder.init(
            g_init_key, 
            np.ones([1, args.goal_end_idx - args.goal_start_idx]),
            input_type="goal"
        )
        
        # Merge params: sa_params has sa_projection, g_params has goal_projection
        merged_params = {**unfreeze(sa_params)['params'], **unfreeze(g_params)['params']}
        u_encoder_params = freeze({'params': merged_params})
        
        critic_state = TrainState.create(
            apply_fn=None,
            params={"u_encoder": u_encoder_params},
            tx=optax.adam(learning_rate=args.critic_lr),
        )
    else:
        sa_encoder = SA_encoder(network_width=args.critic_network_width, network_depth=args.critic_depth, skip_connections=args.critic_skip_connections, use_relu=args.use_relu)
        sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, args.obs_dim]), np.ones([1, action_size]))
        g_encoder = G_encoder(network_width=args.critic_network_width, network_depth=args.critic_depth, skip_connections=args.critic_skip_connections, use_relu=args.use_relu)
        g_encoder_params = g_encoder.init(g_key, np.ones([1, args.goal_end_idx - args.goal_start_idx]))
    
        critic_state = TrainState.create(
            apply_fn=None,
            params={
                "sa_encoder": sa_encoder_params, 
                "g_encoder": g_encoder_params
                },
            tx=optax.adam(learning_rate=args.critic_lr),
        )

    # SIGReg loss params (only needed when using SIGReg/JEPA loss)
    if args.sigreg:
        global sigreg_params
        sigreg_params = SIGRegModule.init_sigreg_params(knots=17)

    # SIGReg params for the actor JEPA loss (separate from critic's sigreg_params)
    # sigreg_params_actor = SIGRegModule.init_sigreg_params(knots=17)

    # Entropy coefficient
    target_entropy = -args.entropy_param * action_size # action_size = 8 for ant, 17 for humanoid, etc
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": log_alpha},
        tx=optax.adam(learning_rate=args.alpha_lr),
    )
    
    # Trainstate
    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
    )

    #Replay Buffer
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

    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        means, _, _ = actor.apply(training_state.actor_state.params, env_state.obs)
        actions = nn.tanh( means )

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
        actions = nn.tanh( means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype) )

        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1-nstate.done,
            extras={"state_extras": state_extras},
        )
        
    def multi_sample_actor_step(training_state, env, env_state, key, K, extra_fields):
        # Get K sets of actions from the actor
        keys = jax.random.split(key, K)
        
        means, log_stds, _ = actor.apply(training_state.actor_state.params, env_state.obs)
        stds = jnp.exp(log_stds)
        
        actions = jnp.stack([
            nn.tanh(means + stds * jax.random.normal(k, shape=means.shape, dtype=means.dtype))
            for k in keys
        ])
        
        state = env_state.obs[:, :args.obs_dim]
        goal = env_state.obs[:, args.obs_dim:]

        if args.unified_encoder:
            # Unified encoder: encode goals and state-actions with same encoder
            g_repr = u_encoder.apply(
                training_state.critic_state.params["u_encoder"], 
                goal,
                input_type="goal"
            )
            sa_reprs = jax.vmap(
                lambda a: u_encoder.apply(
                    training_state.critic_state.params["u_encoder"], 
                    jnp.concatenate([state, a], axis=-1),
                    input_type="state_action"
                )
            )(actions)
            
        else:
            g_repr = g_encoder.apply(
                training_state.critic_state.params["g_encoder"], 
                goal
            ) 
            sa_reprs = jax.vmap(
                lambda a: sa_encoder.apply(
                    training_state.critic_state.params["sa_encoder"], 
                    state, 
                    a
                )
            )(actions)

        q_values = -jnp.sqrt(
            jnp.sum((sa_reprs - g_repr) ** 2, axis=-1)
        )
        
        best_action_idx = jnp.argmax(q_values, axis=0)
        best_actions = jnp.take_along_axis(
            actions,
            best_action_idx[None, :, None],
            axis=0
        )[0]
        
        # Step environment with best actions
        nstate = env.step(env_state, best_actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        
        return nstate, Transition(
            observation=env_state.obs,
            action=best_actions,
            reward=nstate.reward,
            discount=1-nstate.done,
            extras={"state_extras": state_extras},
        )
    
    

    @jax.jit
    def get_experience(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t): #conducts a single actor step in environment
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            if args.expl_actor == 1:
                env_state, transition = actor_step(training_state, env, env_state, current_key, extra_fields=("truncation", "seed"))
            elif args.expl_actor == 0:
                env_state, transition = deterministic_actor_step(training_state, env, env_state, extra_fields=("truncation", "seed"))
            else:
                env_state, transition = multi_sample_actor_step(training_state, env, env_state, current_key, args.expl_actor, extra_fields=("truncation", "seed"))
            return (env_state, next_key), transition

        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=args.unroll_length)

        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state = get_experience(
                training_state,
                env_state,
                buffer_state,
                key,
            
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + args.env_steps_per_actor_step,
            )
            return (training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=args.num_prefill_actor_steps)[0]

    @jax.jit
    def update_actor_and_alpha(transitions, training_state, key):
        actor_batch_size = args.batch_size
        transitions = jax.tree_util.tree_map(
            lambda x: x[:actor_batch_size], 
            transitions
        )
        def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
            obs = transitions.observation           # expected_shape = batch_size, obs_size + goal_size
            state = obs[:, :args.obs_dim]
            future_state = transitions.extras["future_state"]
            goal = future_state[:, args.goal_start_idx : args.goal_end_idx]
            observation = jnp.concatenate([state, goal], axis=1)

            key, sample_key, sig_key1, sig_key2 = jax.random.split(key, 4)

            means, log_stds, actor_embedding = actor.apply(actor_params, observation)
            stds = jnp.exp(log_stds)
            x_ts = means + stds * jax.random.normal(sample_key, shape=means.shape, dtype=means.dtype)
            action = nn.tanh(x_ts)
            log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
            log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
            log_prob = log_prob.sum(-1)           # dimension = B

            if args.unified_encoder:
                # Unified encoder
                sa_repr = u_encoder.apply(
                    critic_params["u_encoder"], 
                    jnp.concatenate([state, action], axis=-1),
                    input_type="state_action"
                )
                g_repr = u_encoder.apply(
                    critic_params["u_encoder"], 
                    goal,
                    input_type="goal"
                )
            else:
                sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]
                sa_repr = sa_encoder.apply(sa_encoder_params, state, action)
                g_repr = g_encoder.apply(g_encoder_params, goal)

            qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))

            # Actor loss: maximise Q (minimise distance)
            actor_loss = -jnp.mean(qf_pi)

            # Actor embedding regularization: diversify backbone representations
            # before the action heads to prevent embedding collapse.
            # iso_actor: push toward isotropic Gaussian N(0,1) (parameter-free)
            # actor_embed_reg: learned SIGReg with sigreg_params_actor
            actor_sigreg_loss = jnp.float32(0.0)
            if args.iso_actor or args.actor_embed_reg:
                if args.iso_actor:
                    actor_sigreg_loss = sigreg_iso(
                        actor_embedding, sig_key1,
                        num_slices=args.sigreg_iso_num_slices,
                        num_t=args.sigreg_iso_num_t,
                        t_max=args.sigreg_iso_t_max,
                    )
                else:
                    actor_sigreg_loss = sigreg_forward(
                        actor_embedding, sigreg_params_actor, sig_key1
                    )
                actor_loss = actor_loss + args.actor_reg_coeff * actor_sigreg_loss

            if not args.disable_entropy:
                actor_loss = actor_loss + jnp.mean(jnp.exp(log_alpha) * log_prob)

            return actor_loss, (log_prob, actor_sigreg_loss)

        def alpha_loss(alpha_params, log_prob):
            '''
            alpha: some term
            alpha loss: alpha * mean(- log prob - entropy) 
            '''
            alpha = jnp.exp(alpha_params["log_alpha"])
            alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - target_entropy))
            return jnp.mean(alpha_loss)
        
        (actorloss, (log_prob, actor_sigreg_loss)), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(training_state.actor_state.params, training_state.critic_state.params, training_state.alpha_state.params['log_alpha'], transitions, key)
        new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

        alphaloss, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
        new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

        training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)

        metrics = {
            "sample_entropy": -log_prob,
            "actor_loss": actorloss,
            "alph_aloss": alphaloss,   
            "log_alpha": training_state.alpha_state.params["log_alpha"],
            "actor_sigreg_loss": actor_sigreg_loss,
        }

        return training_state, metrics

    @jax.jit
    def update_critic(transitions, training_state, key):
        critic_batch_size = args.batch_size
        transitions = jax.tree_util.tree_map(
            lambda x: x[:critic_batch_size], 
            transitions
        )
        def critic_loss(critic_params, transitions, key):
            obs = transitions.observation[:, :args.obs_dim]      
            action = transitions.action                          
            goal = transitions.observation[:, args.obs_dim:]     

            # Step A: Encoder forward — architecture choice
            if args.unified_encoder:
                u_encoder_params = critic_params["u_encoder"]
                sa_repr = u_encoder.apply(
                    u_encoder_params, 
                    jnp.concatenate([obs, action], axis=-1),
                    input_type="state_action"
                )
                g_repr = u_encoder.apply(
                    u_encoder_params, 
                    goal,
                    input_type="goal"
                )
            else:
                sa_repr = sa_encoder.apply(critic_params["sa_encoder"], obs, action)
                g_repr = g_encoder.apply(critic_params["g_encoder"], goal)

            # Step B: Loss computation — loss function choice
            if args.sigreg:
                critic_loss, sim_loss, sigreg_loss = lejepa_loss(
                    g_repr, sa_repr, sigreg_params, lamb=args.lamb, rng=key, M=256
                )
                # InfoNCE
                logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
                critic_loss += -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

                logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
                critic_loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp**2)
                
            else:
                # InfoNCE
                logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
                critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
                
                logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
                critic_loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp**2)
                sim_loss = jnp.zeros(()) 
                sigreg_loss = jnp.zeros(())  

            aux_metrics = {
                'I': jnp.zeros(()),
                'correct': jnp.zeros(()),
                'logits_pos': jnp.zeros(()),
                'logits_neg': jnp.zeros(()),
                'sim_loss' : sim_loss,
                'sigreg_loss': sigreg_loss
            }
            return critic_loss, aux_metrics
        
        (loss, aux_metrics), grad = jax.value_and_grad(
            critic_loss, has_aux=True
        )(training_state.critic_state.params, transitions, key)

        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        training_state = training_state.replace(critic_state=new_critic_state)

        if args.sigreg:
            metrics = {
                "categorical_accuracy": aux_metrics['correct'],
                "logits_pos": aux_metrics['logits_pos'],
                "logits_neg": aux_metrics['logits_neg'],
                # "logsumexp": aux_metrics['logsumexp'],
                'sim_loss': aux_metrics['sim_loss'],
                "critic_loss": loss,
                "sigreg_loss": aux_metrics['sigreg_loss'],
            }
        else:
            metrics = {
                "categorical_accuracy": aux_metrics['correct'],
                "logits_pos": aux_metrics['logits_pos'],
                "logits_neg": aux_metrics['logits_neg'],
                # "logsumexp": aux_metrics['logsumexp'],
                "critic_loss": loss,
            }

        return training_state, metrics

    @jax.jit
    def sgd_step(carry, transitions):
        training_state, key = carry
        key, critic_key, actor_key, = jax.random.split(key, 3)

        training_state, actor_metrics = update_actor_and_alpha(transitions, training_state, actor_key)

        training_state, critic_metrics = update_critic(transitions, training_state, critic_key)

        training_state = training_state.replace(gradient_steps = training_state.gradient_steps + 1)

        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        
        return (training_state, key,), metrics

    @jax.jit
    def training_step(training_state, env_state, buffer_state, key, t):
        experience_key1, experience_key2, sampling_key, training_key, sgd_batches_key = jax.random.split(key, 5)
        
        # update buffer
        env_state, buffer_state = get_experience(
            training_state,
            env_state,
            buffer_state,
            experience_key1,
        )

        training_state = training_state.replace(
            env_steps=training_state.env_steps + args.env_steps_per_actor_step,
        )
            
        transitions_list = []
        for _ in range(args.num_episodes_per_env):
            buffer_state, new_transitions = replay_buffer.sample(buffer_state)
            transitions_list.append(new_transitions)

        # Concatenate all sampled transitions
        transitions = jax.tree_util.tree_map(
            lambda *arrays: jnp.concatenate(arrays, axis=0),
            *transitions_list
        )

        # process transitions for training
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
            (args.gamma, args.obs_dim, args.goal_start_idx, args.goal_end_idx), transitions, batch_keys
        )
        
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )
        
              
        permutation = jax.random.permutation(experience_key2, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        
        num_full_batches = len(transitions.observation) // args.batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[:num_full_batches * args.batch_size], transitions)
        
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]),
            transitions,
        )
        
        if args.use_all_batches == 0:
            num_total_batches = transitions.observation.shape[0]
            selected_indices = jax.random.permutation(
                sgd_batches_key, 
                num_total_batches
            )[:args.num_sgd_batches_per_training_step]
            transitions = jax.tree_util.tree_map(
                lambda x: x[selected_indices], 
                transitions
            )        
        
        # take actor-step worth of training-step
        (training_state, _,), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)

        return (training_state, env_state, buffer_state,), metrics

    @jax.jit
    def training_epoch(
        training_state,
        env_state,
        buffer_state,
        key,
    ):  
        @jax.jit
        def f(carry, t):
            ts, es, bs, k = carry
            k, train_key = jax.random.split(k, 2)
            (ts, es, bs,), metrics = training_step(ts, es, bs, train_key, t)
            return (ts, es, bs, k), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(f, (training_state, env_state, buffer_state, key), jnp.arange(args.num_training_steps_per_epoch * args.training_steps_multiplier))

        
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

    key, prefill_key = jax.random.split(key, 2)

    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key
    )
    

    if args.eval_actor == 0:
        '''Setting up evaluator'''
        evaluator = CrlEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )
        
    elif args.eval_actor == 1:
        key, eval_actor_key = jax.random.split(key)
        evaluator = CrlEvaluator(
            lambda training_state, env, env_state, extra_fields: actor_step(
                training_state,
                env,
                env_state,
                eval_actor_key,
                extra_fields
            ),
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )
    
    elif args.eval_actor > 1:
        key, eval_actor_key = jax.random.split(key)
        evaluator = CrlEvaluator(
            # Replace deterministic_actor_step with a partial function of multi_sample_actor_step
            lambda training_state, env, env_state, extra_fields: multi_sample_actor_step(
                training_state, 
                env, 
                env_state, 
                eval_actor_key, 
                args.eval_actor,
                extra_fields
            ),
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )
    

    training_walltime = 0
    print('starting training....', flush=True)
    start_time = time.time() 
    for ne in range(args.num_epochs):
        
        t = time.time()

        key, epoch_key = jax.random.split(key)
        training_state, env_state, buffer_state, metrics = training_epoch(training_state, env_state, buffer_state, epoch_key)
        
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

        print(f"epoch {ne} out of {args.num_epochs} complete. metrics: {metrics}", flush=True)

        if args.checkpoint:

            if ne < 5 or ne >= args.num_epochs - 5 or ne % 10 == 0:
                # Save current policy and critic params.
                params = (training_state.alpha_state.params, training_state.actor_state.params, training_state.critic_state.params)
                path = f"{save_path}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)
                jax.clear_caches()

        
        if args.track:
            wandb.log(metrics, step=ne)

            if args.wandb_mode == 'offline':
                trigger_sync()
        
        hours_passed = (time.time() - start_time) / 3600
        print(f"Time elapsed: {hours_passed:.3f} hours", flush=True)

    
    if args.checkpoint:
        # Save current policy and critic params.
        params = (training_state.alpha_state.params, training_state.actor_state.params, training_state.critic_state.params)
        path = f"{save_path}/final.pkl"
        save_params(path, params)
        
    # After training is complete, render the final policy
    if args.capture_vis:
        def render_policy(training_state, save_path):
            """Renders the policy and saves it as an HTML file."""
            @jax.jit
            def policy_step(env_state, actor_params):
                means, _, _ = actor.apply(actor_params, env_state.obs)
                actions = nn.tanh(means)
                next_state = env.step(env_state, actions)
                return next_state, env_state 
            
            rollout_states = []
            for i in range(args.num_render):
                env = make_env(args.eval_env_id)
                
                rng = jax.random.PRNGKey(seed=i+1)
                env_state = jax.jit(env.reset)(rng)
                
                for _ in range(args.vis_length):
                    env_state, current_state = policy_step(env_state, training_state.actor_state.params)
                    rollout_states.append(current_state.pipeline_state)
            
            # Render and save
            html_string = html.render(env.sys, rollout_states)
            render_path = f"{save_path}/vis.html"
            with open(render_path, "w") as f:
                f.write(html_string)
            wandb.log({"vis": wandb.Html(html_string)})
            
        print("Rendering final policy...", flush=True)
        try:
            render_policy(training_state, save_path)
        except Exception as e:
            print(f"Error rendering final policy: {e}", flush=True)
        
    #After training is complete, save the Args
    if args.checkpoint:
        with open(f"{save_path}/args.pkl", 'wb') as f:
            pickle.dump(args, f)
        print(f"Saved args to {save_path}/args.pkl", flush=True)
        
    #After training is complete, save the replay buffer (if save_buffer is 1, this takes a lot of memory)
    if args.checkpoint:
        if args.save_buffer:
            print("Saving final buffer_state and buffer data (everything needed to recreate replay_buffer)...", flush=True)
            try:
                buffer_path = f"{save_path}/final_buffer.pkl"
                buffer_data = {
                    'buffer_state': buffer_state,
                    'max_replay_size': args.max_replay_size,
                    'batch_size': args.batch_size,
                    'num_envs': args.num_envs,
                    'episode_length': args.episode_length,
                }
                with open(buffer_path, 'wb') as f:
                    pickle.dump(buffer_data, f)
                print(f"Saved replay_buffer to {buffer_path}", flush=True)
            except Exception as e:
                print(f"Error saving final replay buffer: {e}", flush=True)
