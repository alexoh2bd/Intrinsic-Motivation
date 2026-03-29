"""
Loss Functions 

SIGReg (LeJEPA)
VICReg
INFO_NCE (SimCLR)
"""
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Optional, Tuple
def eu_loss(z_s, z_g):
    """Learned quasimetric in latent space"""
    # Simple L2 for now (can be more sophisticated)
    return jnp.sqrt(jnp.sum((z_s - z_g)**2) + 1e-8)
def tri_loss(z_s, z_next, z_g):
    d_direct = jax.vmap(eu_loss)(z_s, z_g)
    d_s_next = jax.vmap(eu_loss)(z_s, z_next)
    d_next_g = jax.vmap(eu_loss)(z_next, z_g)

    d_indirect = d_s_next + d_next_g

    v = jax.nn.relu(d_direct - d_indirect)
    return jnp.mean(v ** 2)



def lejepa_loss(
    goal_proj: jnp.ndarray,
    sa_proj: jnp.ndarray,
    sigreg_params: dict,
    lamb: float,
    rng: jax.random.PRNGKey,
    M: int = 256
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    goal_proj: (B,D)- Embeddings of global views
    sa_proj: (B, D) - Embeddings of all views (global + local)
    sigreg_params: Dictionary containing SIGReg buffers (t, phi, weights)
    lamb: scalar weight
    """
    # For CRL, we don't have multiple views, so we treat goal and sa as "two views"
    all_proj = jnp.stack([goal_proj, sa_proj], axis=1)  # (B, 2, D)

    # # Prediction loss (MSE between centers and sa pairs)
    # sim_loss = jnp.square(center - sa_reshaped).mean()
    sim_loss = jnp.square(goal_proj -  sa_proj).mean()
    
    # SIGReg loss for each proj
    sigreg_losses = []
    for i in range(all_proj.shape[1]):  # 2 iterations
        view_emb = all_proj[:, i, :]  # (B, D)
        rng, subrng = random.split(rng)
        l = sigreg_forward(view_emb, sigreg_params, subrng, M=M)
        sigreg_losses.append(l)
    
    sigreg_loss = jnp.stack(sigreg_losses).mean()
    
    total_loss = (1 - lamb) * sim_loss + lamb * sigreg_loss
    return total_loss, sim_loss, sigreg_loss

def sigreg_forward(
    proj: jnp.ndarray,
    params: dict,
    rng: jax.random.PRNGKey,
    M: int = 256
) -> jnp.ndarray:
    """
    Forward pass for SIGReg
    
    Args:
        proj: (N, D) - projections
        params: dict with keys 't', 'phi', 'weights'
        rng: random key
        M: number of random projections
    
    Returns:
        scalar loss
    """
    # Generate random matrix A: (D, M)
    A = random.normal(rng, shape=(proj.shape[-1], M), dtype=proj.dtype)
    A = A / jnp.linalg.norm(A, ord=2, axis=0, keepdims=True)
    
    # x_t: (N, M, knots)
    x_t = jnp.expand_dims(proj @ A, axis=-1) * params['t']
    
    # Compute means over batch (dim 0) -> (M, knots)
    cos_mean = jnp.cos(x_t).mean(axis=0)
    sin_mean = jnp.sin(x_t).mean(axis=0)
    
    # Global reduction for distributed training
    # Note: In JAX, you'd use jax.lax.pmean with pmap for multi-device
    # This is a placeholder - actual implementation depends on your parallel strategy
    cos_mean = all_reduce_mean(cos_mean)
    sin_mean = all_reduce_mean(sin_mean)
    
    # Compute error
    err = jnp.square(cos_mean - params['phi']) + jnp.square(sin_mean)
    
    # Weighted sum
    statistic = (err @ params['weights']) * proj.shape[0]
    
    return statistic.mean()




# Distributed training helpers
def all_reduce_mean(x: jnp.ndarray) -> jnp.ndarray:
    """
    All-reduce mean for distributed training
    
    For pmap: use jax.lax.pmean(x, axis_name='devices')
    For single device: just return x
    
    This should be replaced with actual distributed primitives when using pmap/pjit
    """
    # Placeholder - in actual distributed training, use:
    # return jax.lax.pmean(x, axis_name='devices')
    return x


def is_dist_avail_and_initialized() -> bool:
    """Check if distributed training is available"""
    # In JAX, this would check if you're using pmap/pjit
    # Placeholder implementation
    return jax.device_count() > 1


# Flax Module version (alternative implementation)
class SIGRegModule(nn.Module):
    """Flax module version of SIGReg"""
    knots: int = 17
    
    def setup(self):
        # Initialize buffers as Flax variables (not trainable params)
        t = jnp.linspace(0, 3, self.knots, dtype=jnp.float32)
        dt = 3 / (self.knots - 1)
        weights = jnp.full((self.knots,), 2 * dt, dtype=jnp.float32)
        weights = weights.at[0].set(dt)
        weights = weights.at[-1].set(dt)
        window = jnp.exp(-jnp.square(t) / 2.0)
        
        # Register as Flax variables in 'sigreg_buffers' collection
        self.t = self.variable('sigreg_buffers', 't', lambda: t)
        self.phi = self.variable('sigreg_buffers', 'phi', lambda: window)
        self.weights = self.variable('sigreg_buffers', 'weights', lambda: weights * window)

    @staticmethod
    def init_sigreg_params(knots: int = 17) -> dict:
        """Initialize SIGReg parameters (buffers) - static method for direct usage"""
        t = jnp.linspace(0, 3, knots, dtype=jnp.float32)
        dt = 3 / (knots - 1)
        weights = jnp.full((knots,), 2 * dt, dtype=jnp.float32)
        weights = weights.at[0].set(dt)
        weights = weights.at[-1].set(dt)
        window = jnp.exp(-jnp.square(t) / 2.0)
        
        return {
            't': t,
            'phi': window,
            'weights': weights * window
        }
    def __call__(self, proj: jnp.ndarray, rng: jax.random.PRNGKey, M: int = 256) -> jnp.ndarray:
        """
        Args:
            proj: (N, D) - batch of projections
            rng: random key
            M: number of random projections
        
        Returns:
            scalar loss
        """
        # Generate random matrix A: (D, M)
        A = random.normal(rng, shape=(proj.shape[-1], M), dtype=proj.dtype)
        A = A / jnp.linalg.norm(A, ord=2, axis=0, keepdims=True)
        
        # x_t: (N, M, knots)
        x_t = jnp.expand_dims(proj @ A, axis=-1) * self.t
        
        # Compute means over batch (dim 0) -> (M, knots)
        cos_mean = jnp.cos(x_t).mean(axis=0)
        sin_mean = jnp.sin(x_t).mean(axis=0)
        
        # Global reduction for distributed training
        if is_dist_avail_and_initialized():
            cos_mean = all_reduce_mean(cos_mean)
            sin_mean = all_reduce_mean(sin_mean)
        
        # Compute error
        err = jnp.square(cos_mean - self.phi) + jnp.square(sin_mean)
        
        # Weighted sum
        statistic = (err @ self.weights) * proj.shape[0]
        
        return statistic.mean()

