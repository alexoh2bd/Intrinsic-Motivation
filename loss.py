# Compatibility shim — canonical code lives in src/loss.py
from src.loss import *  # noqa: F401,F403
from src.loss import (
    eu_loss,
    tri_loss,
    lejepa_loss,
    sigreg_forward,
    all_reduce_mean,
    is_dist_avail_and_initialized,
    SIGRegModule,
)
