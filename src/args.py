

from dataclasses import dataclass

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

    # PPO / ISOActor specific
    ppo_clip_coef: float = 0.2
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    ppo_epochs: int = 4
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    sigreg_lambda: float = 0.0
    num_slices: int = 16
    num_t: int = 8
    t_max: float = 5.0

