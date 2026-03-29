"""
Network architectures for scaling-CRL.

Contains:
- residual_block: shared residual block used by most networks
- UnifiedEncoder: single encoder for both goals and state-actions (SIGReg path)
- SA_encoder: state-action encoder (InfoNCE path)
- G_encoder: goal encoder (InfoNCE path)
- Actor: deep residual actor
- ShallowActor: 2-layer MLP actor probe
"""
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling
from src.loss import lejepa_loss, SIGRegModule


lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
bias_init = nn.initializers.zeros


def residual_block(x, width, normalize, activation):
    identity = x
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = x + identity
    return x


class UnifiedEncoder(nn.Module):
    """Single encoder for both goals and state-actions"""
    network_width: int = 1024
    network_depth: int = 4
    goal_dim: int = 3
    obs_dim: int = 268
    action_dim: int = 17
    norm_type: str = "layer_norm"
    skip_connections: int = 4
    use_relu: int = 0

    @nn.compact
    def __call__(self, x, input_type: str):
        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        # Input projection - use NAMED layers so both are always created
        goal_proj = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init, name="goal_projection")
        sa_proj = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init, name="sa_projection")

        if input_type == "goal":
            x = goal_proj(x)
        else:  # "state_action"
            x = sa_proj(x)

        x = normalize(x)
        x = activation(x)
        # Shared trunk
        for _ in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation)

        # MLP head
        x = nn.Dense(256, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)


class SA_encoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        x = jnp.concatenate([s, a], axis=-1)
        # Initial layer
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        # Residual blocks
        for i in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation)
        # Final layer
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class G_encoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0

    @nn.compact
    def __call__(self, g: jnp.ndarray):
        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        x = g
        # Initial layer
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        # Residual blocks
        for i in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation)
        # Final layer
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class Actor(nn.Module):
    action_size: int
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 16
    skip_connections: int = 0
    use_relu: int = 0
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5
    sigreg_params = SIGRegModule.init_sigreg_params(knots=17)


    @nn.compact
    def __call__(self, x):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        # Initial layer
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        # Residual blocks
        for i in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation)

        # x is the backbone embedding (exposed for optional regularization)
        embedding = x

        # Output heads
        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)

        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std, embedding


class ShallowActor(nn.Module):
    """
    Variable-depth plain MLP actor probe (no residual connections).
    Used to ablate how much actor capacity is needed when the critic
    representation is frozen.  Sweep num_hidden_layers: 16 → 8 → 4 → 2.
    """
    action_size: int
    hidden_width: int = 256
    num_hidden_layers: int = 2
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        normalize = lambda x: nn.LayerNorm()(x)
        activation = nn.swish

        for _ in range(self.num_hidden_layers):
            x = nn.Dense(self.hidden_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            x = normalize(x)
            x = activation(x)

        # Output heads
        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)

        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)

        return mean, log_std
