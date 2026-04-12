"""
Shared data types for scaling-CRL.
"""
import flax
import jax.numpy as jnp
from typing import NamedTuple
from flax.training.train_state import TrainState


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner"""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState


@flax.struct.dataclass
class ISOTrainingState:
    """Training state for shared-trunk PPO (no separate critic or alpha)."""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState


class Transition(NamedTuple):
    """Container for a transition"""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()
