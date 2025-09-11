from dataclasses import dataclass

@dataclass
class LocalArgs:
    wm_directory: str = "/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/ckpt_data/merged_data/wm_trained_w_merged_data/latest.pt"
    filter_directory_nogp: str = ''
    filter_directory_gp: str = '/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/examples/baselines/dreamerv3-torch/LCRL_SAC/gp/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_1/epoch_id_3/policy.pth'
    offline_data_path: str = '/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/ckpt_data/merged_data/merged_data_rollouts/videos/trajectory_mixed.h5'