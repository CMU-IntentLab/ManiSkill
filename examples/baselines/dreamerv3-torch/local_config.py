from dataclasses import dataclass

@dataclass
class LocalArgs:
    # wm_directory: str = "/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/ckpt_data/aug24_tempckpt/wm_ckpt.pt"
    # filter_directory_nogp: str = ''
    # filter_directory_gp: str = '/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/ckpt_data/aug24_tempckpt/gp_policy.pth'
    # offline_data_path: str = '/home/kensuke/ManiSkill/examples/baselines/ppo/runs/BlockTopple-v0__ppo_rgb__1__1753308792/test_videos/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5'

    # wm_directory: str = "/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/examples/baselines/dreamerv3-torch/runs/BlockTopple-v0__dreamer__1__1756605634/latest.pt"
    # filter_directory_nogp: str = ''
    # filter_directory_gp: str = '/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/examples/baselines/dreamerv3-torch/runs/LCRL/gp/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_1/epoch_id_3/policy.pth'
    # offline_data_path: str = '/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/examples/baselines/ppo/runs/BlockTopple-v0__ppo_rgb__1__1756574937/test_videos/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5'

    wm_directory: str = "/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/ckpt_data/merged_data/wm_trained_w_merged_data/latest.pt"
    filter_directory_nogp: str = ''
    # filter_directory_gp: str = '/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/ckpt_data/merged_data/ddpg_trained_w_merged_data/gp/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_1/epoch_id_3/policy.pth'
    filter_directory_gp: str = '/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/ckpt_data/merged_data/ddpg_trained_w_merged_data/gp/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_1/epoch_id_3/policy.pth'
    # filter_directory_gp: str = '/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/examples/baselines/dreamerv3-torch/LCRL/gp/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_1/epoch_id_3/policy.pth'
    offline_data_path: str = '/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/ckpt_data/merged_data/merged_data_rollouts/videos/trajectory_mixed.h5'
