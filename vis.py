"""Standalone rollout visualiser.

Loads a saved policy (final.pkl / step_*.pkl), rolls out N episodes in Brax,
renders a per-episode GIF via brax.io.image, saves to a gifs/ sub-folder, and
optionally appends the GIFs to an existing or new W&B run.

Usage
-----
# Minimal — infers network type automatically:
    uv run vis.py --params-path runs/humanoid_1000_20250101-120000/final.pkl

# With saved args (recommended — rebuilds the exact network config):
    uv run vis.py \
        --params-path runs/humanoid_1000_20250101-120000/final.pkl \
        --args-path   runs/humanoid_1000_20250101-120000/args.pkl

# Append to an existing W&B run:
    uv run vis.py \
        --params-path  runs/humanoid_1000.../final.pkl \
        --args-path    runs/humanoid_1000.../args.pkl \
        --wandb-run-id <existing-run-id>

# GIF playback defaults to 5× real-time (--playback-speedup 5). Use --fps N for exact frame rate.

Network type detection
-----------------------
trainISO.py  saves actor_state.params directly  → a plain dict pytree → ISOActor
train.py     saves (alpha, actor, critic) params  → a 3-tuple         → Actor
Pass --network-type iso|crl to force, or leave as "auto".
"""

from __future__ import annotations

import gc
import io
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax
import flax.linen as nn
import mujoco
import numpy as np
import wandb
from brax import envs
from PIL import Image

from src.args import Args
from src.env_factory import make_env as _make_env
from src.networks import Actor, ISOActor
from src.utils import load_params


def _patch_mujoco_renderer_close() -> None:
    """MuJoCo Renderer.__init__ can fail after _gl_context but before _mjr_context.

    The stock close()/__del__ then raises AttributeError. Patch close to use getattr.
    """
    _orig = mujoco.Renderer.close

    def _safe_close(self) -> None:
        gl = getattr(self, "_gl_context", None)
        if gl is not None:
            try:
                gl.free()
            except Exception:
                pass
        self._gl_context = None
        mjr = getattr(self, "_mjr_context", None)
        if mjr is not None:
            try:
                mjr.free()
            except Exception:
                pass
        self._mjr_context = None

    mujoco.Renderer.close = _safe_close  # type: ignore[assignment]


_patch_mujoco_renderer_close()


def _gif_frame_duration_ms(system, playback_speedup: float, fps: Optional[int]) -> float:
    """Milliseconds per GIF frame. fps overrides; else dt / playback_speedup."""
    if fps is not None:
        return 1000.0 / float(fps)
    dt_ms = float(np.asarray(system.dt)) * 1000.0
    return dt_ms / float(playback_speedup)


def _render_gif_bytes(
    system,
    trajectory: list,
    height: int,
    width: int,
    camera: Optional[object] = None,
    *,
    playback_speedup: float = 1.0,
    fps: Optional[int] = None,
) -> bytes:
    """Like brax.io.image.render(fmt='gif') but closes Renderer in finally.

    Stock brax leaves Renderer to __del__, which can interact badly with EGL/OSMesa
    teardown (libc++abi recursive init) on GPU/headless clusters.
    """
    if not trajectory:
        raise RuntimeError("must have at least one state")

    renderer = mujoco.Renderer(system.mj_model, height=height, width=width)
    try:
        cam = camera if camera is not None else -1
        frames_np = []
        for state in trajectory:
            d = mujoco.MjData(system.mj_model)
            q = np.asarray(state.q)
            qd = np.asarray(state.qd)
            d.qpos, d.qvel = q, qd
            mujoco.mj_forward(system.mj_model, d)
            renderer.update_scene(d, camera=cam)
            frames_np.append(renderer.render())
    finally:
        try:
            renderer.close()
        except Exception:
            pass

    duration_ms = _gif_frame_duration_ms(system, playback_speedup, fps)

    frames = [Image.fromarray(arr) for arr in frames_np]
    buf = io.BytesIO()
    if len(frames) == 1:
        frames[0].save(buf, format="gif")
    else:
        frames[0].save(
            buf,
            format="gif",
            append_images=frames[1:],
            save_all=True,
            duration=duration_ms,
            loop=0,
        )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI dataclass (parsed by tyro)
# ---------------------------------------------------------------------------

@dataclass
class VisConfig:
    # ── Required ────────────────────────────────────────────────────────────
    params_path: str
    """Path to final.pkl (or step_*.pkl) produced by train.py / trainISO.py."""

    # ── Optional paths ──────────────────────────────────────────────────────
    args_path: Optional[str] = None
    """Path to args.pkl saved alongside the checkpoint (recommended)."""

    output_dir: Optional[str] = None
    """Directory to save GIFs. Defaults to the same directory as params_path."""

    # ── Network / env overrides (used when args.pkl is not available) ────────
    env_id: str = "humanoid"
    network_type: str = "auto"
    """Network type: "auto" | "iso" | "crl". auto infers from params shape."""
    actor_network_width: int = 256
    actor_depth: int = 4
    actor_skip_connections: int = 0
    use_relu: int = 0

    # ── Rollout settings ────────────────────────────────────────────────────
    num_render: int = 5
    """Number of episodes to roll out (one GIF per episode)."""
    vis_length: int = 500
    """Steps per episode."""
    episode_length: int = 1000
    """Max episode length used when wrapping the env."""
    seed: int = 42

    # ── Render settings ─────────────────────────────────────────────────────
    # MuJoCo defaults offscreen framebuffer to 640×480; larger requests raise
    # before Renderer finishes init (and can trigger a noisy __del__ bug).
    width: int = 640
    height: int = 480
    camera: Optional[str] = None
    """Camera name (None = MuJoCo default)."""
    playback_speedup: float = 5.0
    """GIF plays this many times faster than real-time (env dt). Ignored if --fps is set."""
    fps: Optional[int] = None
    """If set, GIF frame duration = 1/fps seconds (overrides playback_speedup)."""

    # ── W&B settings ────────────────────────────────────────────────────────
    wandb_project: str = "JaxGCRL_testing"
    wandb_entity: str = "aho13-duke-university"
    wandb_mode: str = "online"
    wandb_run_id: Optional[str] = None
    """Resume / append to this existing run id. None = create a new run."""
    wandb_run_name: Optional[str] = None
    """Name for a new run (ignored when wandb_run_id is set)."""
    wandb_step: int = 0
    """Step value used when logging vis/* keys."""
    no_wandb: bool = False
    """Skip W&B entirely (just save GIFs locally)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_args_pkl(path: str) -> Args:
    with open(path, "rb") as f:
        return pickle.load(f)


def _detect_network_type(params) -> str:
    """Return 'iso' if params is a plain pytree dict, 'crl' if a 3-tuple."""
    if isinstance(params, tuple) and len(params) == 3:
        return "crl"
    return "iso"


def _extract_actor_params(params, network_type: str):
    """Extract just the actor params from either checkpoint format."""
    if network_type == "crl":
        # (alpha_params, actor_params, critic_params)
        return params[1]
    return params  # ISO: already just actor params


def _build_actor(network_type: str, action_size: int, cfg: VisConfig):
    if network_type == "iso":
        return ISOActor(
            action_size=action_size,
            network_width=cfg.actor_network_width,
            network_depth=cfg.actor_depth,
            skip_connections=cfg.actor_skip_connections,
            use_relu=cfg.use_relu,
        )
    return Actor(
        action_size=action_size,
        network_width=cfg.actor_network_width,
        network_depth=cfg.actor_depth,
        skip_connections=cfg.actor_skip_connections,
        use_relu=cfg.use_relu,
    )


def _make_env_from_cfg(env_id: str, args: Args):
    env = _make_env(env_id, args)
    return envs.training.wrap(env, episode_length=args.episode_length)


def _deterministic_action(actor, actor_params, obs, network_type: str):
    """Return tanh(mean) regardless of actor type."""
    if network_type == "iso":
        means, _, _, _ = actor.apply(actor_params, obs)
    else:
        means, _, _ = actor.apply(actor_params, obs)
    return nn.tanh(means)


def _clamp_render_to_framebuffer(sys, width: int, height: int) -> tuple[int, int]:
    """MuJoCo.Renderer requires width/height ≤ model.vis.global_ offscreen buffer."""
    mj = sys.mj_model
    max_w = int(mj.vis.global_.offwidth)
    max_h = int(mj.vis.global_.offheight)
    w, h = min(width, max_w), min(height, max_h)
    if w != width or h != height:
        print(
            f"  Clamping render size {width}×{height} to framebuffer {max_w}×{max_h} → {w}×{h}",
            flush=True,
        )
    return w, h


# ---------------------------------------------------------------------------
# Core rollout + render
# ---------------------------------------------------------------------------

def render_episodes(
    cfg: VisConfig,
    actor,
    actor_params,
    network_type: str,
    args: Args,
    gif_dir: Path,
) -> list[Path]:
    """Roll out cfg.num_render episodes; save each as a GIF. Return file paths."""

    @jax.jit
    def policy_step(env_state, actor_params):
        actions = _deterministic_action(actor, actor_params, env_state.obs, network_type)
        next_state = env.step(env_state, actions)
        return next_state, env_state

    gif_paths: list[Path] = []

    for ep in range(cfg.num_render):
        env = _make_env_from_cfg(args.eval_env_id or args.env_id, args)
        # Brax training.wrap adds VmapWrapper: reset is vmapped over the leading
        # RNG axis. A bare PRNGKey has shape (2,); vmap would split the key into
        # two scalars. Use split(..., 1) so shape is (1, *key) for a single env.
        rng = jax.random.split(jax.random.PRNGKey(cfg.seed + ep), 1)
        env_state = jax.jit(env.reset)(rng)

        episode_states = []
        for _ in range(cfg.vis_length):
            env_state, captured_state = policy_step(env_state, actor_params)
            episode_states.append(captured_state.pipeline_state)

        print(f"  episode {ep}: rendering {len(episode_states)} frames...", flush=True)

        rw, rh = _clamp_render_to_framebuffer(env.sys, cfg.width, cfg.height)
        render_kwargs: dict = dict(height=rh, width=rw)
        if cfg.camera is not None:
            render_kwargs["camera"] = cfg.camera

        try:
            gif_bytes = _render_gif_bytes(
                env.sys,
                episode_states,
                height=render_kwargs["height"],
                width=render_kwargs["width"],
                camera=render_kwargs.get("camera"),
                playback_speedup=cfg.playback_speedup,
                fps=cfg.fps,
            )
        except Exception as e:
            print(f"  WARNING: GIF render failed for episode {ep}: {e}", flush=True)
            err = str(e).lower()
            if "framebuffer" in err or "offwidth" in err or "offheight" in err:
                print(
                    "  Hint: use --width/--height within the model framebuffer, or add "
                    "<visual><global offwidth=\"...\" offheight=\"...\"/></visual> to the env XML.",
                    flush=True,
                )
            else:
                print(
                    "  Hint: headless/GPU nodes: try export MUJOCO_GL=egl (GPU) or "
                    "MUJOCO_GL=osmesa (CPU offscreen); ensure libGL/OSMesa/EGL is installed.",
                    flush=True,
                )
            continue

        gif_path = gif_dir / f"episode_{ep:03d}.gif"
        gif_path.write_bytes(gif_bytes)
        print(f"  saved → {gif_path}", flush=True)
        gif_paths.append(gif_path)
        gc.collect()

    return gif_paths


# ---------------------------------------------------------------------------
# W&B logging
# ---------------------------------------------------------------------------

def log_to_wandb(cfg: VisConfig, gif_paths: list[Path], params_path: Path):
    """Create or resume a W&B run and log each GIF as wandb.Video."""
    init_kwargs: dict = dict(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        mode=cfg.wandb_mode,
    )
    if cfg.wandb_run_id:
        init_kwargs["id"] = cfg.wandb_run_id
        init_kwargs["resume"] = "allow"
        print(f"Resuming W&B run {cfg.wandb_run_id}", flush=True)
    else:
        run_name = cfg.wandb_run_name or f"vis_{params_path.parent.name}"
        init_kwargs["name"] = run_name
        print(f"Creating new W&B run: {run_name}", flush=True)

    run = wandb.init(**init_kwargs)

    log_dict = {}
    for gif_path in gif_paths:
        key = f"vis/{gif_path.stem}"
        log_dict[key] = wandb.Video(str(gif_path), format="gif")

    wandb.log(log_dict, step=cfg.wandb_step)
    print(f"Logged {len(log_dict)} GIF(s) to W&B run {run.id}", flush=True)
    wandb.finish()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    import tyro
    cfg = tyro.cli(VisConfig)

    params_path = Path(cfg.params_path)
    if not params_path.exists():
        print(f"ERROR: params file not found: {params_path}", file=sys.stderr)
        sys.exit(1)

    # ── Output directory ────────────────────────────────────────────────────
    out_dir = Path(cfg.output_dir) if cfg.output_dir else params_path.parent
    gif_dir = out_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    print(f"GIFs will be saved to: {gif_dir}", flush=True)

    # ── Load params ─────────────────────────────────────────────────────────
    print(f"Loading params from {params_path}", flush=True)
    params = load_params(str(params_path))

    # ── Detect or override network type ─────────────────────────────────────
    if cfg.network_type == "auto":
        network_type = _detect_network_type(params)
        print(f"Auto-detected network type: {network_type}", flush=True)
    else:
        network_type = cfg.network_type  # "iso" or "crl"

    actor_params = _extract_actor_params(params, network_type)

    # ── Reconstruct Args (for env / network shapes) ──────────────────────────
    if cfg.args_path:
        args = _load_args_pkl(cfg.args_path)
        print(f"Loaded training Args from {cfg.args_path}", flush=True)
        # Allow CLI overrides for render-specific fields
        args.num_render = cfg.num_render
        args.vis_length = cfg.vis_length
    else:
        print("No args.pkl provided — using VisConfig defaults for network / env.", flush=True)
        args = Args(
            env_id=cfg.env_id,
            eval_env_id=cfg.env_id,
            episode_length=cfg.episode_length,
            actor_network_width=cfg.actor_network_width,
            actor_depth=cfg.actor_depth,
            actor_skip_connections=cfg.actor_skip_connections,
            use_relu=cfg.use_relu,
            num_render=cfg.num_render,
            vis_length=cfg.vis_length,
        )

    # ── Build env + actor ────────────────────────────────────────────────────
    dummy_env = _make_env_from_cfg(args.eval_env_id or args.env_id, args)
    action_size = dummy_env.action_size
    obs_size = dummy_env.observation_size

    actor = _build_actor(network_type, action_size, cfg if not cfg.args_path else _cfg_from_args(args))
    print(
        f"Actor: {actor.__class__.__name__}  obs={obs_size}  act={action_size}  "
        f"width={actor.network_width}  depth={actor.network_depth}",
        flush=True,
    )

    # Quick param shape sanity-check (forward pass on dummy input)
    try:
        dummy_obs = np.ones([1, obs_size], dtype=np.float32)
        actor.apply(actor_params, dummy_obs)
        print("Param sanity check passed.", flush=True)
    except Exception as e:
        print(f"WARNING: sanity check failed ({e}). Proceeding anyway.", flush=True)

    # ── Render episodes ──────────────────────────────────────────────────────
    print(f"\nRendering {cfg.num_render} episodes × {cfg.vis_length} steps...", flush=True)
    gif_paths = render_episodes(cfg, actor, actor_params, network_type, args, gif_dir)

    if not gif_paths:
        print("No GIFs produced (rendering failed for all episodes). Exiting.", flush=True)
        sys.exit(1)

    # ── W&B ──────────────────────────────────────────────────────────────────
    if not cfg.no_wandb:
        log_to_wandb(cfg, gif_paths, params_path)
    else:
        print("W&B logging skipped (--no-wandb).", flush=True)

    print(f"\nDone. {len(gif_paths)} GIF(s) saved to {gif_dir}", flush=True)


def _cfg_from_args(args: Args) -> "VisConfig":
    """Minimal VisConfig-like object that carries network shape fields from Args."""
    cfg = VisConfig.__new__(VisConfig)
    cfg.actor_network_width = args.actor_network_width
    cfg.actor_depth = args.actor_depth
    cfg.actor_skip_connections = args.actor_skip_connections
    cfg.use_relu = args.use_relu
    cfg.camera = None
    return cfg


if __name__ == "__main__":
    main()
