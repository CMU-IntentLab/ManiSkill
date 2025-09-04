import argparse
import os
import sys

import gymnasium #as gym
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# dreamer_dir = os.path.abspath('/home/kensuke/WM_CBF/ManiSkill/examples/baselines/dreamerv3-torch')
dreamer_dir = os.path.abspath('/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/examples/baselines/dreamerv3-torch')
sys.path.append(dreamer_dir)

# maniskill_dir = os.path.abspath('/home/kensuke/WM_CBF/ManiSkill/mani_skill')
maniskill_dir = os.path.abspath('/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/mani_skill')
sys.path.append(maniskill_dir)


pyHJ_dir = os.path.abspath('/home/clown2/Desktop/Work/Research/ManiSkill/PytorchReachability')
sys.path.append(pyHJ_dir)

import models
import tools
import ruamel.yaml as yaml

from PyHJ.data import Collector, VectorReplayBuffer
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.trainer import offpolicy_trainer
from PyHJ.utils import TensorboardLogger, WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
import PyHJ.reach_rl_gym_envs as reach_rl_gym_envs

from termcolor import cprint
from datetime import datetime
import pathlib
from pathlib import Path
import collections
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import tyro

# note: need to include the dreamerv3 repo for this
from dreamer import make_dataset
from config import Args
# NOTE: all the reach-avoid gym environments are in reach_rl_gym, the constraint information is output as an element of the info dictionary in gym.step() function
"""
    Note that, we can pass arguments to the script by using
    python run_training_ddpg.py --task ra_droneracing_Game-v6 --control-net 512 512 512 512 --disturbance-net 512 512 512 512 --critic-net 512 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9
    python run_training_ddpg.py --task ra_highway_Game-v2 --control-net 512 512 512 --disturbance-net 512 512 512 --critic-net 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9
    python run_training_ddpg.py --task ra_1d_Game-v0 --control-net 32 32 --disturbance-net 4 4 --critic-net 4 4 --epoch 10 --total-episodes 160 --gamma 0.9
    
    For learning the classical reach-avoid value function (baseline):
    python run_training_ddpg.py --task ra_droneracing_Game-v6 --control-net 512 512 512 512 --disturbance-net 512 512 512 512 --critic-net 512 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9 --is-game-baseline True
    python run_training_ddpg.py --task ra_highway_Game-v2 --control-net 512 512 512 --disturbance-net 512 512 512 --critic-net 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9 --is-game-baseline True
    python run_training_ddpg.py --task ra_1d_Game-v0 --control-net 32 32 --disturbance-net 4 4 --critic-net 4 4 --epoch 10 --total-episodes 160 --gamma 0.9 --is-game-baseline True

"""

def main(args):

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    config = args


    image_size = 128
    cam_obs_space = gym.spaces.Box(
            low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
        )
    policy_obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

    bool_space = gym.spaces.Box(low=0, high=1, shape=(), dtype=bool)

    observation_space = gym.spaces.Dict({
            'front_cam': cam_obs_space,
            'is_first': bool_space,
            'is_last': bool_space,
            'is_terminal': bool_space,
            'state': policy_obs_space,
            'wrist_cam': cam_obs_space,
        })
    action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)


    config.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]

    wm = models.WorldModel(observation_space, action_space, 0, config)

    # checkpoint = torch.load("/home/kensuke/WM_CBF/ManiSkill/examples/baselines/dreamerv3-torch/runs/wm_edit/wm_lz.pt")
    # checkpoint = torch.load("/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/examples/baselines/dreamerv3-torch/runs/BlockTopple-v0__dreamer__1__1756605634/latest.pt")
    checkpoint = torch.load("/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/ckpt_data/merged_data/wm_trained_w_merged_data/latest.pt")

    state_dict = {k[14:]:v for k,v in checkpoint['agent_state_dict'].items() if '_wm' in k}

    wm.load_state_dict(state_dict)
    wm.eval()


    config.batch_size = 1
    config.batch_length = 5
    expert_eps = collections.OrderedDict()
    # config.offline_data_path = '/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/examples/baselines/ppo/runs/BlockTopple-v0__ppo_rgb__1__1756574937/test_videos/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5'
    config.offline_data_path = '/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/ckpt_data/merged_data/merged_data_rollouts/videos/trajectory_mixed.h5'
    tools.fill_expert_dataset(config, expert_eps)
    expert_dataset = make_dataset(expert_eps, config)


    # NOTE: you can replace this with the dataset you made for the dubins wm training
    # directory = '/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/examples/baselines/dreamerv3-torch/runs/BlockTopple-v0__dreamer__1__1756605634/train_eps'
    directory = '/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/ckpt_data/merged_data/wm_trained_w_merged_data/train_eps'
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    train_dataset = make_dataset(train_eps, config)

    # NOTE: should only need 1 dataset: the offline dataset u collected from the script.
    datasets = [train_dataset, expert_dataset]

    env = gymnasium.make('BlockToppleWM-v0', params = [wm, datasets, config])


    # check if the environment has control and disturbance actions:
    assert hasattr(env, 'action_space') #and hasattr(env, 'action2_space'), "The environment does not have control and disturbance actions!"
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    args.max_action = env.action_space.high[0]

    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]

    train_envs = DummyVectorEnv(
        [lambda: gymnasium.make('BlockToppleWM-v0', params = [wm, datasets, config]) for _ in range(args.training_num)]
    )
    test_envs = DummyVectorEnv(
        [lambda: gymnasium.make('BlockToppleWM-v0', params = [wm, datasets, config]) for _ in range(args.test_num)]
    )


    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model

    if args.actor_activation == 'ReLU':
        actor_activation = torch.nn.ReLU
    else:
        raise ValueError("Please provide actor_activation!")

    if args.critic_activation == 'ReLU':
        critic_activation = torch.nn.ReLU
    else:
        raise ValueError("Please provide critic_activation!")

    if args.critic_net is not None:
        critic_net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.critic_net,
            activation=critic_activation,
            concat=True,
            device=args.device
        )
    else:
        # report error:
        raise ValueError("Please provide critic_net!")

    critic = Critic(critic_net, device=args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    log_path = None

    # from PyHJ.policy import avoid_DDPGPolicy_annealing_acreg as DDPGPolicy
    from PyHJ.policy.modelfree.ddpg_avoid_classical_acreg import avoid_DDPGPolicy_annealing_acreg as DDPGPolicy

    print("DDPG under the Avoid annealed Bellman equation with no Disturbance has been loaded!")

    actor_net = Net(args.state_shape, hidden_sizes=args.control_net, activation=actor_activation, device=args.device)
    actor = Actor(
        actor_net, args.action_shape, max_action=args.max_action, device=args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)


    policy = DDPGPolicy(
    critic,
    critic_optim,
    tau=args.tau,
    gamma=args.gamma_pyhj,
    exploration_noise=GaussianNoise(sigma=args.exploration_noise),
    reward_normalization=args.rew_norm,
    estimation_step=args.n_step,
    action_space=env.action_space,
    actor=actor,
    actor_optim=actor_optim,
    actor_gradient_steps=args.actor_gradient_steps,
    )

    if args.use_gp:
        log_path = os.path.join('LCRL/gp')
    else:
        log_path = os.path.join('LCRL/no_gp')
    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs)

    if args.warm_start_path is not None:
        policy.load_state_dict(torch.load(args.warm_start_path))
        args.kwargs = args.kwargs + "warmstarted"

    epoch = 0
    # writer = SummaryWriter(log_path, filename_suffix="_"+timestr+"epoch_id_{}".format(epoch))
    # logger = TensorboardLogger(writer)
    log_path = log_path+'/noise_{}_actor_lr_{}_critic_lr_{}_batch_{}_step_per_epoch_{}_kwargs_{}_seed_{}'.format(
            args.exploration_noise, 
            args.actor_lr, 
            args.critic_lr, 
            args.batch_size_pyhj,
            args.step_per_epoch,
            args.kwargs,
            args.seed
        )


    if args.continue_training_epoch is not None:
        epoch = args.continue_training_epoch
        policy.load_state_dict(torch.load(
            os.path.join(
                log_path+"/epoch_id_{}".format(epoch),
                "policy.pth"
            )
        ))


    if args.continue_training_logdir is not None:
        policy.load_state_dict(torch.load(args.continue_training_logdir))
        # epoch = int(args.continue_training_logdir.split('_')[-9].split('_')[0])
        epoch = args.continue_training_epoch


    def save_best_fn(policy, epoch=epoch):
        torch.save(
            policy.state_dict(), 
            os.path.join(
                log_path+"/epoch_id_{}".format(epoch),
                "policy.pth"
            )
        )


    def stop_fn(mean_rewards):
        return False

    if not os.path.exists(log_path+"/epoch_id_{}".format(epoch)):
        print("Just created the log directory!")
        # print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
        os.makedirs(log_path+"/epoch_id_{}".format(epoch))

    warmup = 1
    logger = None

    for iter in range(warmup+args.total_episodes):
        if iter  < warmup:
            policy._gamma = 0 # for warmup the value fn
            policy.warmup = True
            steps_per_collect = 8
            steps = 40000
        else:
            # policy._gamma = config.gamma_pyhj
            policy._gamma = 0.99
            policy.warmup = False
            steps_per_collect = 8
            steps = 40000

        if args.continue_training_epoch is not None:
            print("epoch: {}, remaining epochs: {}".format(epoch//args.epoch, args.total_episodes - iter))
        else:
            print("epoch: {}, remaining epochs: {}".format(iter, args.total_episodes - iter))
        epoch = epoch + args.epoch
        print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
        if args.total_episodes > 1:
            writer = SummaryWriter(log_path+"/epoch_id_{}".format(epoch)) #filename_suffix="_"+timestr+"_epoch_id_{}".format(epoch))
        else:
            if not os.path.exists(log_path+"/total_epochs_{}".format(epoch)):
                print("Just created the log directory!")
                print("log_path: ", log_path+"/total_epochs_{}".format(epoch))
                os.makedirs(log_path+"/total_epochs_{}".format(epoch))
            writer = SummaryWriter(log_path+"/total_epochs_{}".format(epoch)) #filename_suffix="_"+timestr+"_epoch_id_{}".format(epoch))
        if logger is None:
            logger = WandbLogger()
            logger.load(writer)
        logger = TensorboardLogger(writer)
        
        # import pdb; pdb.set_trace()
        result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size_pyhj,
        update_per_step=args.update_per_step,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger
        )
        
        save_best_fn(policy, epoch=epoch)


if __name__ == "__main__":
    args = tyro.cli(Args)

    main(args)
