# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.dreamer_wrapper import DreamerWrapper, SelectAction, UUID

import pathlib
import tools
import collections  
import models
import functools
from torch import distributions as torchd
from typing import Union

import numpy as np
import sapien
import torch
from tqdm.auto import tqdm
from transforms3d.quaternions import quat2axangle

from mani_skill.agents.controllers import (
    PDEEPosController,
    PDEEPoseController,
    PDJointPosController,
    PDJointVelController,
)
from mani_skill.agents.controllers.base_controller import CombinedController
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.pose import Pose
from mani_skill.trajectory.utils.actions import conversion
to_np = lambda x: x.detach().cpu().numpy()

from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix, matrix_to_euler_angles
import numpy as np
import torch




@dataclass
class Args:
    exp_name: Optional[str] = None
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
    save_trajectory: bool = False
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
    num_envs: int = 16
    """the number of parallel environments"""
    num_eval_envs: int = 10
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
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    render_mode: str = "all"
    """the environment rendering mode"""

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    buffer_device: str = "cuda:0"
    """where the replay buffer is stored. Can be 'cpu' or 'cuda' for GPU"""
    gamma: float = 0.8
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient"""
    batch_size: int = 512
    """the batch size of sample from the replay memory"""
    learning_starts: int = 4_000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    training_freq: int = 64
    """training frequency (in steps)"""
    utd: float = 0.25
    """update to data ratio"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    bootstrap_at_done: str = "always"
    """the bootstrap method to use when a done signal is received. Can be 'always' or 'never'"""
    camera_width: Optional[int] = None
    """the width of the camera image. If none it will use the default the environment specifies"""
    camera_height: Optional[int] = None
    """the height of the camera image. If none it will use the default the environment specifies."""

    # to be filled in runtime
    grad_steps_per_iteration: int = 0
    """the number of gradient updates per iteration"""
    steps_per_env: int = 0
    """the number of steps each parallel env takes per iteration"""


    parallel: bool = True
    eval_every: int = 1e4
    eval_episode_num: int = 10
    log_every: int = 1e4
    reset_every: int =  0
    device: str = 'cuda:0'
    compile: bool = True
    precision: int =  32
    debug: bool =  False
    video_pred_log: bool =  True
    precision: int = 32
    action_repeat: int = 2
    steps = int = 1e6

    eval_every: int = 1e4
    log_every: int = 1e4
    time_limit: int = 1e3
    offline_traindir: str = ''
    offline_evaldir: str = ''
    reset_every: int = 0

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
    batch_length: int = 64
    train_ratio: int = 512
    pretrain: int = 100
    model_lr: float = 1e-4
    opt_eps: float = 1e-8
    grad_clip: int = 1000
    dataset_size: int = 1000000
    opt: str = 'adam'

    # Environment
    #task: 'dmc_walker_walk'
    #size: [64, 64]
    #envs: 1
    action_repeat: int = 2
    time_limit: int = 1000
    grayscale: bool = False
    prefill: int = 2500
    reward_EMA: bool = True

    # Behavior.
    discount: float = 0.997
    discount_lambda: float = 0.95
    imag_horizon: int = 15
    imag_gradient: str = 'dynamics'
    imag_gradient_mix: float =  0.0
    eval_state_mean: bool = False
        

def convert_joint_delta_actions_to_ee_delta_actions(
    joint_delta_actions: np.ndarray,  # shape: (T, num_joints)
    controller_from: PDJointPosController,
    controller_to: Union[PDEEPoseController, PDEEPosController],
    articulation,  # usually env.agent.robot
    pos_only: bool = False,
) -> np.ndarray:
    """
    Convert a sequence of pd_joint_delta_pos actions to pd_ee_delta_pose or pd_ee_delta_pos.
    Returns a (T, 6) or (T, 3) array of normalized actions.
    """
    assert controller_from.config.use_delta and controller_from.config.normalize_action
    assert controller_to.config.use_delta and controller_to.config.normalize_action

    low, high = controller_from.config.lower, controller_from.config.upper
    delta_actions = []
    
    qpos = controller_from.qpos.clone()

    pin_model = articulation.create_pinocchio_model()
    ee_index = controller_to.ee_link.index

    for t in range(len(joint_delta_actions)):
        delta_q = gym_utils.clip_and_scale_action(joint_delta_actions[t], low, high)
        qpos = qpos + delta_q
        pin_model.compute_forward_kinematics(qpos.cpu().numpy()[0])
        ee_pose = Pose.create(articulation.pose.sp * pin_model.get_link_pose(ee_index))

        if t == 0:
            prev_pose = ee_pose
            continue  # need two poses to compute delta

        # Compute delta pose
        delta_position = ee_pose.p - prev_pose.p
        if pos_only:
            delta = delta_position.cpu().numpy()[0]
        else:
            delta_q = (prev_pose.sp * ee_pose.sp.inv()).q
            delta = np.r_[
                delta_position.cpu().numpy()[0],
                conversion.compact_axis_angle_from_quaternion(delta_q),
            ]

        # Normalize
        low2 = controller_to.action_space_low.cpu().numpy()
        high2 = controller_to.action_space_high.cpu().numpy()
        delta = gym_utils.inv_scale_action(delta, low2, high2)
        delta_actions.append(delta)

        prev_pose = ee_pose

    return np.stack(delta_actions)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    print('made device')
    ####### Environment setup #######
    env_kwargs = dict(obs_mode=args.obs_mode, render_mode=args.render_mode, sim_backend="gpu", sensor_configs=dict())
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    if args.camera_width is not None:
        # this overrides every sensor used for observation generation
        env_kwargs["sensor_configs"]["width"] = args.camera_width
    if args.camera_height is not None:
        env_kwargs["sensor_configs"]["height"] = args.camera_height
    print('about to make')
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    
    data = np.load('/home/kensuke/ManiSkill/examples/baselines/dreamerv3-torch/runs/BlockTopple-v0_base-dreamer/eval_eps/0bfe0300-c359-4e2b-a862-e6521c727117-100.npz')
    joint_ac = data['action']
    ee_ac = convert_joint_delta_actions_to_ee_delta_actions(joint_ac, PDJointPosController, PDEEPosController,
    envs.agent.robot,  # usually env.agent.robot
    pos_only= False)
    print(ee_ac)
    print('made envs')