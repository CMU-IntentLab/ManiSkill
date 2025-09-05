from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Args:
    exp_name: Optional[str] = "FilterRollout"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "dreamer"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_trajectory: bool = True
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    log_freq: int = 1_000_000
    """logging frequency in terms of environment steps"""

    # Environment specific arguments
    env_id: str = "BlockTopple-v0"
    """the id of the environment"""
    obs_mode: str = "rgb"
    """the observation mode to use"""
    include_state: bool = True
    """whether to include the state in the observation"""
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    num_envs: int = 1
    """the number of parallel environments"""
    num_eval_envs: int = 1
    """the number of parallel evaluation environments"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    eval_freq: int = 1_000_000
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = "pd_ee_delta_pose"
    """the control mode to use for the environment"""
    render_mode: str = "all"
    """the environment rendering mode"""

    
    camera_width: Optional[int] = None
    """the width of the camera image. If none it will use the default the environment specifies"""
    camera_height: Optional[int] = None
    """the height of the camera image. If none it will use the default the environment specifies."""

    # to be filled in runtime
    """the number of gradient updates per iteration"""
    steps_per_env: int = 0
    """the number of steps each parallel env takes per iteration"""


    parallel: bool = True
    eval_every: int = 10_000
    eval_episode_num: int = 10
    log_every: int = 10_000
    reset_every: int =  0
    device: str = 'cuda:0'
    compile: bool = True
    precision: int =  32
    debug: bool =  False
    video_pred_log: bool =  True
    action_repeat: int = 1
    steps: int = 10_000_000

    #time_limit: int = 1e3
    offline_traindir: str = ''
    offline_evaldir: str = ''

    dyn_hidden: int = 512
    dyn_deter: int = 512
    dyn_stoch: int = 32
    dyn_discrete: int = 32
    dyn_rec_depth: int = 1
    dyn_mean_act: str = 'none'
    dyn_std_act: str = 'sigmoid2'
    dyn_min_std: float = 0.1
    units: int = 512
    act: str ='SiLU'
    norm: bool = True
    dyn_scale: float = 0.5
    rep_scale: float = 0.1
    kl_free: float = 1.0
    weight_decay: float = 0.0
    unimix_ratio: float = 0.01
    initial: str = 'learned'



    # Exploration
    expl_behavior: str = 'greedy'
    expl_until: int = 1000
    expl_extr_scale: float = 0.0
    expl_intr_scale: float = 1.0
    disag_target: str = 'stoch'
    disag_log: bool =True
    disag_models: int = 10
    disag_offset: int = 1
    disag_layers: int = 4
    disag_units: int = 400
    disag_action_cond: bool = False


    batch_size: int = 32
    batch_length: int = 16
    train_ratio: int = 64
    model_lr: float = 1e-4
    opt_eps: float = 1e-8
    grad_clip: int = 1000
    dataset_size: int = 1_000_000
    opt: str = 'adam'


    time_limit: int = 100
    grayscale: bool = False
    prefill: int = 2500
    reward_EMA: bool = True

    # Behavior.
    discount: float = 0.997
    discount_lambda: float = 0.95
    imag_horizon: int = 15
    imag_gradient: str = 'dynamics'
    imag_gradient_mix: float =  0.0
    eval_state_mean: bool = True


    encoder: Dict[str, Any] = field(default_factory=lambda:{'mlp_keys': 'state', 'cnn_keys': '.*\_cam$', 'act': 'SiLU', 'norm': True, 'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5, 'mlp_units': 1024, 'symlog_inputs': True})
    decoder: Dict[str, Any] = field(default_factory=lambda:{'mlp_keys': 'state', 'cnn_keys': '.*\_cam$', 'act': 'SiLU', 'norm': True, 'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5, 'mlp_units': 1024, 'cnn_sigmoid': False, 'image_dist': 'mse', 'vector_dist': 'symlog_mse', 'outscale': 1.0})
    actor: Dict[str, Any] = field(default_factory=lambda:{'layers': 2, 'dist': 'normal', 'entropy': 3e-4, 'unimix_ratio': 0.01, 'std': 'learned', 'min_std': 0.1, 'max_std': 1.0, 'temp': 0.1, 'lr': 3e-5, 'eps': 1e-5, 'grad_clip': 100.0, 'outscale': 1.0})
    critic: Dict[str, Any] = field(default_factory=lambda:{'layers': 2, 'dist': 'symlog_disc', 'slow_target': True, 'slow_target_update': 1, 'slow_target_fraction': 0.02, 'lr': 3e-5, 'eps': 1e-5, 'grad_clip': 100.0, 'outscale': 0.0})
    reward_head:  Dict[str, Any] = field(default_factory=lambda:{'layers': 2, 'dist': 'symlog_disc', 'loss_scale': 1.0, 'outscale': 0.0})
    cont_head:  Dict[str, Any] = field(default_factory=lambda:{'layers': 2, 'loss_scale': 1.0, 'outscale': 1.0})
    margin_head:  Dict[str, Any] = field(default_factory=lambda:{'layers': 2, 'loss_scale': 1.0})
    grad_heads: List[str] = field(default_factory=lambda: ['decoder', 'reward', 'cont'])



    reward_threshold: Optional[float] = None
    buffer_size: int = 40000
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma_pyhj: float = 0.9999 # type=float, default=0.95)
    tau: float = 0.005 # type=float, default=0.005)
    exploration_noise: float = 0.1 # type=float, default=0.1)
    epoch: int = 1 # type=int, default=10)
    total_episodes: int = 20 # type=int, default=160)
    step_per_epoch: int = 40000 # type=int, default=40000)
    step_per_collect: int = 8 # type=int, default=8)
    update_per_step: float = 0.125 # type=float, default=0.125)
    batch_size_pyhj: int = 512 # type=int, default=512)
    control_net: List[int] = field(default_factory=lambda: [ 512, 512, 512]) # type=int, nargs="*", default=None) # for control policy
    critic_net: List[int] = field(default_factory=lambda: [512, 512, 512])  # type=int, nargs="*", default=None) # for critic net
    training_num: int = 1 # type=int, default=8)
    test_num: int = 1 # type=int, default=100)
    render: float = 0. # type=float, default=0.)
    rew_norm: bool = False # action="store_true", default=False)
    n_step: int = 1 # type=int, default=1)
    continue_training_logdir: Optional[str] = None # type=str, default=None)
    continue_training_epoch: Optional[int] = None # type=int, default=None)
    actor_gradient_steps: int = 1 # type=int, default=1)
    is_game_baseline: bool = False # type=bool, default=False) # it will be set automatically
    target_update_freq: int = 400 # type=int, default=400)
    auto_alpha: float = 1
    alpha_lr: float = 3e-4
    alpha: float = 0.2
    weight_decay_pyhj: float = 0.001
    actor_activation: str = "ReLU" #type=str, default="ReLU")
    critic_activation: str = "ReLU"
    warm_start_path: Optional[str] = None # type=str, default=None)
    kwargs: Dict[str, Any] = field(default_factory=lambda: {}) # type=str, default="")

    gamma_lx: float = 0.75
    offline_data_path: str = '/home/kensuke/ManiSkill/examples/baselines/ppo/runs/BlockTopple-v0__ppo_rgb__1__1753308792/test_videos/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5'
    pretrain: int = 500
    hybrid_steps: int = 1_000_000
    hybrid: bool = True


    use_gp: bool = True
    wm_directory: str = "/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/kens_ckpts/wm_lz_aug6.pt"
    filter_directory_nogp: str = '/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/kens_ckpts/no_gp_aug6.pth'
    filter_directory_gp: str = '/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/kens_ckpts/gp_aug6.pth'
    filter_thresh: float = 0.2
    num_runs: int = 1
    cbf_gamma: float = 0.7
    filter_mode: str = 'cbf' # 'cbf' or 'least_restrictive' or 'lr' or 'none'
    policy: str = 'ppo' # 'ppo' or 'mpc'