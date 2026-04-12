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
from src.args import Args
from brax import envs
from typing import NamedTuple
from wandb_osh.hooks import TriggerWandbSyncHook
from flax.training.train_state import TrainState
from brax.io import html
from src.types import ISOTrainingState, Transition
from src.utils import load_params, save_params
from src.env_factory import make_env as _make_env
from src.loss import sigreg_iso
from src.evaluator import CrlEvaluator
from src.networks import ISOActor
# from train import get_experience,

class PPORollout(NamedTuple):
    """On-policy rollout storage for PPO."""
    observation: jnp.ndarray
    action: jnp.ndarray
    action_pretanh: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


def metrics_to_wandb(d):
    """Convert JAX/NumPy leaves to Python types so wandb.log serializes reliably."""
    if isinstance(d, dict):
        return {k: metrics_to_wandb(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return type(d)(metrics_to_wandb(x) for x in d)
    if isinstance(d, (int, float, bool, str, type(None))):
        return d
    if isinstance(d, wandb.Html):
        return d
    x = jax.device_get(d)
    if isinstance(x, np.generic):
        return float(x)
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x.item())
        return x.tolist()
    if hasattr(x, "item") and getattr(x, "ndim", None) == 0:
        try:
            return float(x.item())
        except Exception:
            pass
    if isinstance(x, (float, np.floating)):
        return float(x)
    return x


if __name__ == "__main__":

    args = tyro.cli(Args)

    print("Arguments:", flush=True)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}", flush=True)
    print("\n", flush=True)

    args.env_steps_per_actor_step = args.num_envs * args.unroll_length
    print(f"env_steps_per_actor_step: {args.env_steps_per_actor_step}", flush=True)

    num_updates = args.total_env_steps // args.env_steps_per_actor_step
    args.num_training_steps_per_epoch = num_updates // args.num_epochs
    print(f"num_updates: {num_updates}", flush=True)
    print(f"num_training_steps_per_epoch: {args.num_training_steps_per_epoch}", flush=True)

    run_name = (
        f"ISO_{args.env_id}_ppo_{args.batch_size}_{args.total_env_steps}"
        f"_nenvs:{args.num_envs}_width:{args.actor_network_width}"
        f"_depth:{args.actor_depth}_sigreg:{args.sigreg_lambda}_{args.seed}"
    )
    print(f"run_name: {run_name}", flush=True)

    trigger_sync = None
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
        trigger_sync = None
        if args.wandb_mode == 'offline':
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()
        print(
            f"wandb: mode={args.wandb_mode} run_dir={wandb.run.dir}",
            flush=True,
        )
        if args.wandb_mode == 'offline':
            print(
                "wandb offline: sync this run with "
                f"wandb sync {wandb.run.dir}",
                flush=True,
            )

    save_path = None
    if args.checkpoint or args.capture_vis:
        from pathlib import Path
        from datetime import datetime
        short_run_name = f"runs/{args.env_id}_{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        save_path = Path(args.wandb_dir) / Path(short_run_name)
        save_path.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, env_key, eval_env_key, actor_key = jax.random.split(key, 4)

    # ---------- Environment ----------
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

    # ---------- Network ----------
    actor = ISOActor(
        action_size=action_size,
        network_width=args.actor_network_width,
        network_depth=args.actor_depth,
        skip_connections=args.actor_skip_connections,
        use_relu=args.use_relu,
    )

    total_gradient_steps = num_updates * args.ppo_epochs * (
        (args.unroll_length * args.num_envs) // args.batch_size
    )
    if args.anneal_lr:
        lr_schedule = optax.linear_schedule(
            init_value=args.actor_lr,
            end_value=0.0,
            transition_steps=total_gradient_steps,
        )
    else:
        lr_schedule = args.actor_lr

    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, np.ones([1, obs_size])),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(learning_rate=lr_schedule, eps=1e-5),
        ),
    )

    training_state = ISOTrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
    )

    # ---------- Actor step helpers (for evaluation) ----------
    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        means, _, _, _ = actor.apply(training_state.actor_state.params, env_state.obs)
        actions = nn.tanh(means)
        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )

    def stochastic_actor_step(training_state, env, env_state, key, extra_fields):
        means, log_stds, _, _ = actor.apply(training_state.actor_state.params, env_state.obs)
        stds = jnp.exp(log_stds)
        actions = nn.tanh(means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype))
        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )

    # ---------- PPO: collect on-policy rollout ----------
    @jax.jit
    def collect_rollout(training_state, env_state, key):
        def step_fn(carry, _):
            env_state, current_key = carry
            current_key, sample_key = jax.random.split(current_key)

            means, log_stds, values, _ = actor.apply(
                training_state.actor_state.params, env_state.obs
            )
            stds = jnp.exp(log_stds)
            x_t = means + stds * jax.random.normal(sample_key, shape=means.shape)
            actions = nn.tanh(x_t)

            log_prob = jax.scipy.stats.norm.logpdf(x_t, loc=means, scale=stds)
            log_prob = log_prob - jnp.log(1 - jnp.square(actions) + 1e-6)
            log_prob = log_prob.sum(axis=-1)

            next_env_state = env.step(env_state, actions)

            rollout = PPORollout(
                observation=env_state.obs,
                action=actions,
                action_pretanh=x_t,
                log_prob=log_prob,
                value=values,
                reward=next_env_state.reward,
                done=next_env_state.done,
            )
            return (next_env_state, current_key), rollout

        (env_state, _), rollout = jax.lax.scan(
            step_fn, (env_state, key), None, length=args.unroll_length
        )

        _, _, last_value, _ = actor.apply(
            training_state.actor_state.params, env_state.obs
        )
        return env_state, rollout, last_value

    # ---------- PPO: GAE ----------
    def compute_gae(rollout, last_value):
        """Generalized Advantage Estimation (reverse scan)."""
        def gae_step(carry, t):
            next_value, next_advantage = carry
            done = rollout.done[t]
            value = rollout.value[t]
            reward = rollout.reward[t]
            delta = reward + args.gamma * next_value * (1 - done) - value
            advantage = delta + args.gamma * args.gae_lambda * (1 - done) * next_advantage
            return (value, advantage), advantage

        _, advantages = jax.lax.scan(
            gae_step,
            (last_value, jnp.zeros_like(last_value)),
            jnp.arange(args.unroll_length - 1, -1, -1),
        )
        advantages = advantages[::-1]
        returns = advantages + rollout.value
        return advantages, returns

    # ---------- PPO: update ----------
    @jax.jit
    def ppo_update(training_state, rollout, advantages, returns, key):
        batch_size = args.unroll_length * args.num_envs
        num_minibatches = batch_size // args.batch_size

        flat_obs = rollout.observation.reshape(batch_size, -1)
        flat_actions = rollout.action.reshape(batch_size, -1)
        flat_pretanh = rollout.action_pretanh.reshape(batch_size, -1)
        flat_log_probs = rollout.log_prob.reshape(batch_size)
        flat_values = rollout.value.reshape(batch_size)
        flat_advantages = advantages.reshape(batch_size)
        flat_returns = returns.reshape(batch_size)

        def loss_fn(params, mb_obs, mb_pretanh, mb_actions, mb_old_log_prob,
                     mb_old_values, mb_adv, mb_ret, rng):
            means, log_stds, values, trunk = actor.apply(params, mb_obs)
            stds = jnp.exp(log_stds)

            log_prob = jax.scipy.stats.norm.logpdf(mb_pretanh, loc=means, scale=stds)
            log_prob = log_prob - jnp.log(1 - jnp.square(mb_actions) + 1e-6)
            log_prob = log_prob.sum(axis=-1)

            ratio = jnp.exp(log_prob - mb_old_log_prob)

            # Clipped surrogate
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * jnp.clip(ratio, 1 - args.ppo_clip_coef, 1 + args.ppo_clip_coef)
            pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

            # Clipped value loss
            v_loss_unclipped = jnp.square(values - mb_ret)
            v_clipped = mb_old_values + jnp.clip(
                values - mb_old_values, -args.ppo_clip_coef, args.ppo_clip_coef
            )
            v_loss_clipped = jnp.square(v_clipped - mb_ret)
            v_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_unclipped, v_loss_clipped))

            # Gaussian entropy: sum of 0.5 * log(2*pi*e*sigma^2) over action dims
            entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * jnp.square(stds))
            entropy = entropy.sum(axis=-1).mean()

            sigreg_loss = jnp.float32(0.0)
            if args.sigreg_lambda > 0:
                sigreg_loss = sigreg_iso(
                    trunk, rng,
                    num_slices=args.num_slices,
                    num_t=args.num_t,
                    t_max=args.t_max,
                )

            total_loss = (
                pg_loss
                + args.vf_coef * v_loss
                - args.ent_coef * entropy
                + args.sigreg_lambda * sigreg_loss
            )
            return total_loss, {
                "pg_loss": pg_loss,
                "v_loss": v_loss,
                "entropy": entropy,
                "sigreg_loss": sigreg_loss,
                "approx_kl": jnp.mean((ratio - 1) - jnp.log(ratio)),
            }

        def epoch_step(carry, _):
            training_state, key = carry
            key, perm_key, loss_key = jax.random.split(key, 3)

            perm = jax.random.permutation(perm_key, batch_size)
            shuffled = (
                flat_obs[perm],
                flat_pretanh[perm],
                flat_actions[perm],
                flat_log_probs[perm],
                flat_values[perm],
                flat_advantages[perm],
                flat_returns[perm],
            )
            truncated = jax.tree_util.tree_map(
                lambda x: x[:num_minibatches * args.batch_size],
                shuffled,
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape(num_minibatches, args.batch_size, *x.shape[1:]),
                truncated,
            )

            def minibatch_step(carry, mb):
                training_state, key = carry
                key, sig_key = jax.random.split(key)
                mb_obs, mb_pretanh, mb_actions, mb_lp, mb_val, mb_adv, mb_ret = mb

                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    training_state.actor_state.params,
                    mb_obs, mb_pretanh, mb_actions, mb_lp, mb_val, mb_adv, mb_ret,
                    sig_key,
                )
                new_actor_state = training_state.actor_state.apply_gradients(grads=grads)
                training_state = training_state.replace(
                    actor_state=new_actor_state,
                    gradient_steps=training_state.gradient_steps + 1,
                )
                return (training_state, key), (loss, aux)

            (training_state, key), (losses, auxs) = jax.lax.scan(
                minibatch_step, (training_state, loss_key), minibatches
            )
            return (training_state, key), (losses, auxs)

        (training_state, _), (all_losses, all_auxs) = jax.lax.scan(
            epoch_step, (training_state, key), None, length=args.ppo_epochs
        )

        metrics = jax.tree_util.tree_map(jnp.mean, all_auxs)
        metrics["total_loss"] = jnp.mean(all_losses)
        return training_state, metrics

    # ---------- Training step / epoch ----------
    @jax.jit
    def training_step(training_state, env_state, key):
        key, rollout_key, update_key = jax.random.split(key, 3)

        env_state, rollout, last_value = collect_rollout(
            training_state, env_state, rollout_key
        )
        advantages, returns = compute_gae(rollout, last_value)
        training_state, metrics = ppo_update(
            training_state, rollout, advantages, returns, update_key
        )
        training_state = training_state.replace(
            env_steps=training_state.env_steps + args.env_steps_per_actor_step,
        )
        return training_state, env_state, metrics

    @jax.jit
    def training_epoch(training_state, env_state, key):
        def f(carry, _):
            ts, es, k = carry
            k, step_key = jax.random.split(k)
            ts, es, metrics = training_step(ts, es, step_key)
            return (ts, es, k), metrics

        (training_state, env_state, _), metrics = jax.lax.scan(
            f, (training_state, env_state, key),
            None, length=args.num_training_steps_per_epoch,
        )
        return training_state, env_state, metrics

    # ---------- Evaluator ----------
    if args.eval_actor == 0:
        evaluator = CrlEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )
    else:
        key, eval_actor_key = jax.random.split(key)
        evaluator = CrlEvaluator(
            lambda training_state, env, env_state, extra_fields: stochastic_actor_step(
                training_state, env, env_state, eval_actor_key, extra_fields
            ),
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )

    # ---------- Main loop ----------
    training_walltime = 0
    print('starting training....', flush=True)
    start_time = time.time()

    for ne in range(args.num_epochs):
        t = time.time()

        key, epoch_key = jax.random.split(key)
        training_state, env_state, metrics = training_epoch(
            training_state, env_state, epoch_key
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

        print(f"epoch {ne} out of {args.num_epochs} complete. metrics: {metrics}", flush=True)

        if args.checkpoint:
            if ne < 5 or ne >= args.num_epochs - 5 or ne % 10 == 0:
                params = training_state.actor_state.params
                path = f"{save_path}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)
                jax.clear_caches()

        if args.track:
            wandb.log(metrics_to_wandb(metrics), step=ne)
            if args.wandb_mode == 'offline' and trigger_sync is not None:
                trigger_sync()

        hours_passed = (time.time() - start_time) / 3600
        print(f"Time elapsed: {hours_passed:.3f} hours", flush=True)

    if args.checkpoint:
        params = training_state.actor_state.params
        path = f"{save_path}/final.pkl"
        save_params(path, params)

    if args.capture_vis and save_path is not None:
        def render_policy(training_state, save_path_local):
            """Roll out the policy in fresh eval envs and save Brax HTML (same pattern as train.py)."""
            @jax.jit
            def policy_step(env_state, actor_params):
                means, _, _, _ = actor.apply(actor_params, env_state.obs)
                actions = nn.tanh(means)
                # Must use `env` from render_policy's loop below — not the training `env` in module scope.
                next_state = env.step(env_state, actions)
                return next_state, env_state

            rollout_states = []
            for i in range(args.num_render):
                env = make_env(args.eval_env_id)
                rng = jax.random.PRNGKey(seed=i + 1)
                env_state = jax.jit(env.reset)(rng)
                for _ in range(args.vis_length):
                    env_state, current_state = policy_step(
                        env_state, training_state.actor_state.params
                    )
                    rollout_states.append(current_state.pipeline_state)

            html_string = html.render(env.sys, rollout_states)
            render_path = save_path_local / "vis.html"
            with open(render_path, "w") as f:
                f.write(html_string)
            print(f"Saved rollout HTML to {render_path}", flush=True)
            if args.track:
                wandb.log(
                    {"vis/policy": wandb.Html(html_string)},
                    step=args.num_epochs,
                )
                if args.wandb_mode == "offline" and trigger_sync is not None:
                    trigger_sync()

        print("Rendering final policy (vis.html)...", flush=True)
        try:
            render_policy(training_state, save_path)
        except Exception as e:
            print(f"Error rendering final policy: {e}", flush=True)

    if args.checkpoint:
        with open(f"{save_path}/args.pkl", 'wb') as f:
            pickle.dump(args, f)
        print(f"Saved args to {save_path}/args.pkl", flush=True)

    if args.track:
        wandb.finish()
