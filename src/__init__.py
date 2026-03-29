# src/ package — shared modules for scaling-crl training scripts
from src.networks import (
    lecun_unfirom,
    bias_init,
    residual_block,
    UnifiedEncoder,
    SA_encoder,
    G_encoder,
    Actor,
    ShallowActor,
)
from src.types import TrainingState, Transition
from src.utils import load_params, save_params
from src.env_factory import make_env
from src.loss import (
    eu_loss,
    tri_loss,
    lejepa_loss,
    sigreg_forward,
    all_reduce_mean,
    is_dist_avail_and_initialized,
    SIGRegModule,
)
from src.buffer import ReplayBufferState, TrajectoryUniformSamplingQueue
from src.evaluator import generate_unroll, CrlEvaluator
