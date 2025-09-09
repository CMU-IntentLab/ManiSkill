# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter # type: ignore
import matplotlib.pyplot as plt
from itertools import product
import imageio.v2 as imageio
from mpl_toolkits.mplot3d import Axes3D
from basic_mpc import EndEffectorMPC

# ManiSkill specific imports
import h5py
import json
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
from gymnasium import spaces

from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.exploration import GaussianNoise
from PyHJ.data import Batch


from config import Args
from local_config import LocalArgs
import wandb

if __name__ == "__main__":
    singularities = np.array([0, 7, 41, 50, 57, 60, 76, 85, 88, 89, 92, 93, 99, 100, 111, 116, 119, 125, 133, 145, 146, 150, 152, 174, 175, 190, 
                    220, 225, 234, 236, 238, 241, 252, 261, 278, 279, 281, 283, 290, 296, 297, 301, 306, 312, 313, 326, 347, 354, 369, 
                    370, 409, 427, 429, 437, 440, 442, 453, 457, 473, 475, 489, 498, 504, 555, 556, 558, 560, 581, 583, 584, 585, 592, 
                    594, 611, 613, 623, 635, 640, 647, 656, 680, 689, 696, 700, 713, 718, 721, 734, 744, 747, 761, 766, 790, 794, 810, 
                    815, 835, 851, 853, 855, 856, 862, 865, 875, 882, 890, 892, 900, 910, 916, 923, 937, 939, 948, 956, 960, 964, 979, 
                    982, 989, 994])

    # Cut from h5
    src_file = '/home/wmcbf/ManiSkill/runs/mpc_data_v1/videos/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5'
    dst_file = os.path.splitext(src_file)[0] + "_filtered.h5"

    with h5py.File(src_file, "r") as f_in, h5py.File(dst_file, "w") as f_out:
        for k in range(len(f_in.keys())):
            if k in singularities:
                continue
            f_in.copy("traj_" + str(k), f_out)

    # Cut from JSON
    src_file = '/home/wmcbf/ManiSkill/runs/mpc_data_v1/videos/trajectory.rgb.pd_ee_delta_pose.physx_cuda.json'
    dst_file = os.path.splitext(src_file)[0] + "_filtered.json"

    with open(src_file, "r") as f:
        data = json.load(f)

    data['episodes'] = [ep for ep in data['episodes'] if ep['episode_id'] not in singularities]

    # 💾 Save to new file
    with open(dst_file, "w") as f:
        json.dump(dst_file, f, indent=2)
