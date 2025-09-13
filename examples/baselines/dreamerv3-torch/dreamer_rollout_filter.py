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
import pickle

# ManiSkill specific imports
import h5py
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

to_np = lambda x: x.detach().cpu().numpy()

os.environ["WANDB_MODE"] = "disabled"

def combine_dictionaries(
    one_dict: Dict[str, Any], other_dict: Dict[str, Any], take_half: bool = False
) -> Dict[str, Any]:
    """
    Combine two dictionaries by interleaving their values.

    Args:
        one_dict (Dict[str, Any]): The first dictionary.
        other_dict (Dict[str, Any]): The second dictionary.
        take_half (bool, optional): Whether to only take the first half of the values. Defaults to False.
    """
    combined = {}
    unused_keys = set(one_dict.keys()) - set(other_dict.keys())
    assert set(unused_keys).issubset(
        {"logprob", "object_state", "privileged_state", "env_ids", "success"}
    ), f"Missing {unused_keys}"
    for k, v in one_dict.items():
        if k in unused_keys:
            continue
        if isinstance(v, dict):
            combined[k] = combine_dictionaries(v, other_dict[k], take_half)
        elif v is None or v.shape[0] == 0:
            combined[k] = other_dict[k]
        elif other_dict[k] is None or other_dict[k].shape[0] == 0:
            combined[k] = v
        else:
            if take_half:
                half_index = v.shape[0] // 2
                v = v[:half_index]
                other_v = other_dict[k][:half_index]
            else:
                other_v = other_dict[k]

            tmp = np.empty((v.shape[0] + other_v.shape[0], *v.shape[1:]), dtype=v.dtype)
            tmp[0::2] = v
            tmp[1::2] = other_v
            combined[k] = tmp

    return combined

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, args):
    generator = tools.sample_episodes(episodes, args.batch_length)
    dataset = tools.from_generator(generator, args.batch_size)
    return dataset
    

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()

class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, args, logger, dataset, expert_dataset=None):
        super(Dreamer, self).__init__()
        self._args = args
        self._logger = logger
        self._should_log = tools.Every(args.log_every)
        batch_steps = args.batch_size * args.batch_length
        self._should_train = tools.Every(batch_steps / args.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(args.reset_every)
        self._should_expl = tools.Until(int(args.expl_until / args.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // args.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._expert_dataset = expert_dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, args)
        self._task_behavior = models.ImagBehavior(args, self._wm)
        if (
            args.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(args, act_space),
            plan2explore=lambda: expl.Plan2Explore(args, self._wm, reward),
        )[args.expl_behavior]().to(self._args.device)
        self.hybrid = args.hybrid
        self.gp_metrics = np.array([0,0,0,0])
        self.nogp_metrics = np.array([0,0,0,0])

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._args.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                if self.hybrid and step < self._args.hybrid_steps:
                    learner_data, exp_data = (
                            next(self._dataset),
                            next(self._expert_dataset),
                        )
                    self._train(learner_data, expert_data=exp_data)
                else:
                    self._train(next(self._dataset), expert_data=None)
                #self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count

            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._args.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._args.action_repeat * self._step

        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        # latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"], sample=False)

        do_sample = True
        # TODO: add logic for continuous latents
        if self._args.eval_state_mean:
            # latent["stoch"] = latent["mean"]
            do_sample = False
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"], sample=do_sample)
        feat = self._wm.dynamics.get_feat(latent)

        gp = self._wm.heads["margin_gp"](feat)[0].item()
        no_gp = self._wm.heads["margin_nogp"](feat)[0].item()

        '''
        # metrics are TN, FP, TP, FN
        if obs['failure'].item() == 1 and gp < 0: # TN
            self.gp_metrics += np.array([1, 0, 0, 0])
        if obs['failure'].item() == 1 and no_gp < 0:
            self.nogp_metrics += np.array([1, 0, 0, 0])
        if obs['failure'].item() == 1 and gp > 0: # FP
            self.gp_metrics += np.array([0, 1, 0, 0])
        if obs['failure'].item() == 1 and no_gp > 0:
            self.nogp_metrics += np.array([0, 1, 0, 0])

        if obs['failure'].item() == 0 and gp < 0: # FN
            self.gp_metrics += np.array([0, 0, 0, 1])
        if obs['failure'].item() == 0 and no_gp < 0:
            self.nogp_metrics += np.array([0, 0, 0, 1])
        if obs['failure'].item() == 0 and gp > 0: # TP
            self.gp_metrics += np.array([0, 0, 1, 0])
        if obs['failure'].item() == 0 and no_gp > 0:
            self.nogp_metrics += np.array([0, 0, 1, 0])'''

        #print('fail', obs['failure'].item(), 'gp', gp, 'no_gp', no_gp)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()

        action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._args.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._args.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data, expert_data):
        metrics = {}
        if self.hybrid and self._step < self._args.hybrid_steps:
            mixed_data = combine_dictionaries(data, expert_data, take_half=True)
            post, context, mets = self._wm._train(mixed_data)
        else:
            post, context, mets = self._wm._train(data)
        metrics.update(mets)


        start = self._wm._get_post(data) #post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._args.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)



def V(state, policy):
    tmp_obs = np.array(state)#.reshape(1,-1)
    tmp_batch = Batch(obs = tmp_obs, info = Batch())
    ac = policy(tmp_batch, model="actor_old").act
    tmp = policy.critic(tmp_batch.obs, ac)
    return tmp.cpu().detach().numpy().flatten()

def Q_v1(state, action, policy):
    if isinstance(action, dict):
        action = action['action']
    b, _ = action.shape
    if state.shape[0] != b:
        state = np.repeat(state, repeats=b, axis=0)
    tmp_obs = np.array(state)#.reshape(1,-1)
    tmp_batch = Batch(obs = tmp_obs, info = Batch())
    tmp = policy.critic(tmp_batch.obs, action)
    return tmp.cpu().detach().numpy().flatten()

def Q_v2(latent, action, policy, agent):
    if isinstance(action, dict):
        action = action['action']
    action = torch.tensor(action, dtype=torch.float32).to(latent['stoch'].device)

    # batch computation
    b, _ = action.shape
    if latent['stoch'].shape[0] != b:
        latent = {
            k: v.repeat(b, *([1] * (v.ndim - 1)))
            for k, v in latent.items()
        }

    # model-based rollout
    # img_latent = agent._wm.dynamics.img_step(latent, action)
    do_sample = True
    if agent._args.eval_state_mean:
        # img_latent["stoch"] = img_latent["mean"]
        do_sample = False
    img_latent = agent._wm.dynamics.img_step(latent, action, sample=do_sample)
    img_feat = agent._wm.dynamics.get_feat(img_latent).cpu().detach().numpy()
    return V(img_feat, policy)

def Qfn(agent_state, actions, agent, safe_policy):
    feat = agent._wm.dynamics.get_feat(agent_state).cpu().detach().numpy()
    q1 = Q_v1(feat, actions, safe_policy)
    q2 = Q_v2(agent_state, actions, safe_policy, agent)
    return np.minimum(q1, q2)

def pi_safe(state, policy):
    tmp_obs = np.array(state)#.reshape(1,-1)
    tmp_batch = Batch(obs = tmp_obs, info = Batch())
    return policy(tmp_batch, model="actor_old").act.cpu().detach().numpy()#.flatten()

def get_ee_pose(envs):
    env_agent = envs.unwrapped.agent
    ee_pose = env_agent.robot.links_map[env_agent.ee_link_name].pose.raw_pose.cpu().detach().numpy()
    return ee_pose

def get_block_pose(envs):
    env_block = envs.unwrapped.block3
    block_pose = env_block.pose.raw_pose.cpu().detach().numpy()
    return block_pose

def steer_to_pose(envs, action, des_pose):
    ee_pose = get_ee_pose(envs)
    err = des_pose - ee_pose[0, :3]
    action['action'] = 0*action['action']
    action['action'][0, :3] = np.clip(err, -0.2, 0.2)
    action['action'][0, 6] = 0.5
    return action

def pi_test(envs, action): # Move the gripper over the green block for testing the filter with a simple policy
    env_agent = envs.unwrapped.agent
    ee_pose = env_agent.robot.links_map[env_agent.ee_link_name].pose.raw_pose.cpu().detach().numpy()
    env_block = envs.unwrapped.block3
    block_pose = env_block.pose.raw_pose.cpu().detach().numpy()
    err = block_pose[0, :3] - ee_pose[0, :3]
    err[2] += 0.1
    err[0] -= 0.035
    err[1] -= 0.05
    action['action'] = 0*action['action']
    action['action'][0, :3] = np.clip(10*err, -0.2, 0.2)
    action['action'][0, 6] = 0.5
    return action

def pi_sweep(envs, action):
    action['action'] = 0*action['action']
    action['action'][0, 1] = 0.0
    action['action'][0, 6] = 0.5
    return action

# Sweeps the actions from an initial condition
def sweep_action(envs, action, i):
    # envs.reset()
    state = envs.unwrapped.get_state().detach().clone()
    action['action'] = 0*action['action']
    N = 100
    delta_ee = np.zeros((10, N))
    for (k, u) in enumerate(np.linspace(-1, 1, N)):
        start_pose = get_ee_pose(envs)
        action['action'][0, i] = u
        for j in range(10):
            envs.step(action)
            delta_pose = get_ee_pose(envs) - start_pose
            delta_ee[j, k] = delta_pose[0, i]
        envs.unwrapped.set_state(state)
        print(k)
    return delta_ee

def bang_bang_test(envs, action, u, i):
    envs.reset()
    state = envs.unwrapped.get_state().detach().clone()
    action['action'] = 0*action['action']
    action['action'][0, i] = u
    N = 10
    delta_ee = np.zeros((2, 10))
    start_pose = get_ee_pose(envs)
    for k in range(N):
        envs.step(action)
        delta_pose = get_ee_pose(envs) - start_pose
        if k > N/2:
            action['action'][0, i] = 0
        delta_ee[0, k] = delta_pose[0, i]
        delta_ee[1, k] = action['action'][0, i]/5
        print(k)
    return delta_ee

import imageio
from PIL import Image
class SamplingVideo():
    def __init__(self):
        self.frames = []
        self.max_ac = np.array([0.76098621, 0.30531207, 0.34810847, 0.0697008,  0.14093682, 0.0133229, 0.59313494])
        self.min_ac = np.array([-0.1864568, -0.22532985, -0.25439265, -0.10240789, -0.09638732, -0.12006265, -1.53002357])

    def add_frame(self, _wm, state, obs_vec, action, Q):
        N = 128*128
        data = _wm.preprocess(obs_vec)
        embed = _wm.encoder(data).unsqueeze(0).repeat(N, 1, 1)
        truth_wrist = data["wrist_cam"].squeeze()
        truth_front = data["front_cam"].squeeze()
        is_first = data["is_first"].unsqueeze(0).repeat(N, 1, 1)

        action_range = torch.linspace(-1, 1, 128)

        actions = torch.stack([torch.tensor([x, y, 0, 0, 0, 0, 0]) for x in action_range for y in action_range])
        actions[:-1, -1] = float(action[0, -1]) # Override gripper with nominal

        qvals = (torch.tensor(Q(state, to_np(actions))) + 1)/2
        
        actions = actions.unsqueeze(1).to(embed.device)

        state_repeat = {'logit': state['logit'].repeat(N, 1, 1, 1),
                 'deter': state['deter'].repeat(N, 1, 1),
                 'stoch': state['stoch'].repeat(N, 1, 1, 1)}

        states, _ = _wm.dynamics.observe(embed, actions, is_first)#, state=state_repeat)#, state=state_repeat)

        l_gp = torch.tanh(_wm.heads['margin_gp'](_wm.dynamics.get_feat(states))).squeeze().reshape(128, 128, 1).repeat(1, 1, 3)
        l_gp = (l_gp + 1)/2
        qvals = qvals.squeeze().reshape(128, 128, 1).repeat(1, 1, 3)

        self.frames.append(torch.cat([torch.cat([truth_wrist.detach().cpu(), truth_front.detach().cpu()], 1), 
                        torch.cat([l_gp.detach().cpu(), qvals], 1)], 0))

    def save(self, filepath):
        imageio.mimsave(filepath, self.frames, fps=2)
        self.frames = []

def rollout_policy(
    nom_policy,
    safe_policy,
    agent,
    envs,
    num_trajs=0,
    thresh=0.6,
    cbf_gamma = 0.7,
    filter_mode='least_restrictive', # 'cbf' or 'least_restrictive' or 'none'
    policy='ppo', # 'ppo' or 'mpc'
    sampling_video=False
):
    torch.cuda.empty_cache()
    
    episode = 0
    
    obs_vec, info = envs.reset()
    obs_vec['failure'] = info['is_knocked_over']
    done_vec = np.zeros(envs.num_envs, bool)

    agent_state = None

    if policy == 'mpc':
        mpc = EndEffectorMPC(get_block_pose(envs)[0, :3], 10)
    elif policy == 'sweep': # Use MPC to do a sweep
        mpc = Sweep(get_block_pose(envs)[0, :3],)

    # get output dir for plots (there is likely a better way to do this)
    output_dir = envs._env.env.env.output_dir.with_name('figs')
    output_dir.mkdir(parents=True, exist_ok=True) 
    
    # statistics from the offline dataset
    max_ac = np.array([0.76098621, 0.30531207, 0.34810847, 0.0697008,  0.14093682, 0.0133229, 0.59313494])
    min_ac = np.array([-0.1864568, -0.22532985, -0.25439265, -0.10240789, -0.09638732, -0.12006265, -1.53002357])

    # Record outcomes, values, actions
    knocked_over = False
    successes = 0
    sample_vals = [[] for _ in range(num_trajs)]
    safe_vals = [[] for _ in range(num_trajs)]
    taken_vals = [[] for _ in range(num_trajs)]
    ee_trajs = [[] for _ in range(num_trajs)]
    nominal_actions = [[] for _ in range(num_trajs)]
    safe_actions = [[] for _ in range(num_trajs)]
    sample_actions = [[] for _ in range(num_trajs)]
    taken_actions = [[] for _ in range(num_trajs)]
    outcomes = {"fail":[],"grasped":[],"lifted":[],"success":[]}

    # MAIN ENV STEP LOOP
    Q = functools.partial(Qfn, agent=agent, safe_policy=safe_policy)
    ac_prev = None

    if sampling_video:
        video = SamplingVideo()
        video_dir = envs._env.env.env.output_dir.with_name('samp_vids')
        video_dir.mkdir(parents=True, exist_ok=True)

    while episode < num_trajs:
        # Update nominal Dreamer policy (also updates latent using observation)
        action, agent_state = nom_policy(obs_vec, done_vec, agent_state)
        state = agent_state[0].copy()

        # Features
        feat = agent._wm.dynamics.get_feat(agent_state[0]).cpu().detach().numpy()

        # Margin function
        l_gp = torch.tanh(agent._wm.heads['margin_gp'](agent._wm.dynamics.get_feat(agent_state[0])))
        l_nogp = torch.tanh(agent._wm.heads['margin_nogp'](agent._wm.dynamics.get_feat(agent_state[0])))

        # action is a dict with keys action and logprob        
        if isinstance(action, dict):
            action = {k: np.array(action[k].detach().cpu()) for k in action}
        else:
            action = np.array(action)

        # Query MPC policy
        if policy == 'mpc':
            mpc.get_action(get_ee_pose(envs)[0, :3], action)

        # Get safe policy
        ac_safe_norm = pi_safe(feat, safe_policy)

        # Normalized and unnormalized safe and nominal action (env uses unnormalized, Q uses normalized)
        ac_unnorm = action['action']
        ac_norm = (action['action'] - min_ac) / (max_ac - min_ac) * 2 - 1
        ac_safe = (ac_safe_norm + 1) * 0.5 * (max_ac - min_ac) + min_ac

        # Create interpolation coefficients: shape (N_interp, 1) for filtering
        N_interp = 10
        t = torch.linspace(0, 1, steps=N_interp).unsqueeze(1)  # shape (N_interp, 1)
        ac_norms = (1 - t) * ac_norm + t * ac_safe_norm

        # # Other sampling methods
        # # ac_unnorms = torch.from_numpy(np.row_stack([np.linspace(ac_unnorm.flatten(), ac_safe.flatten(), 20), 
        # #                            np.linspace(ac_unnorm.flatten(), 0*ac_unnorm.flatten(), 20), 
        # #                            np.linspace(0*ac_unnorm.flatten(), ac_safe.flatten(), 20)]))
        
        # # # ac_unnorms = torch.tensor(list(product(torch.linspace(-1, 1, 5).tolist(), repeat=6)))
        # # # ac_unnorms = torch.cat([ac_unnorms, torch.zeros(ac_unnorms.shape[0], 1)], dim=1)
        # # # ac_unnorms = torch.cat([ac_unnorms, torch.from_numpy(ac_unnorm), torch.from_numpy(ac_safe)], dim=0)
        # # ac_norms = (ac_unnorms - min_ac) / (max_ac - min_ac) * 2 - 1
        ac_norms[:-1, -1] = ac_norm[0, -1] # Override gripper with nominal

        # Get qvals
        qvals = Q(state, ac_norms)
        val = qvals[-1] # Safe policy
        qval = qvals[0]
        # print('V:',val)

        # Record for plotting
        sample_vals[episode].append(qvals)
        safe_vals[episode].append(val)
        ee_trajs[episode].append(get_ee_pose(envs)[0, :3])
        nominal_actions[episode].append(ac_unnorm)
        safe_actions[episode].append(ac_safe)
        sample_actions[episode].append((ac_norms + 1) * 0.5 * (max_ac - min_ac) + min_ac)
        
        # Filter (assumes that ac_norms[0] is nominal, ac_norms[-1] is safe, all the others are samples)
        if filter_mode == 'cbf':
            cbf_thresh = max(cbf_gamma * max(val - thresh, 0), thresh)
            valid_actions = (qvals >= cbf_thresh).astype(bool)

            if np.any(valid_actions):
                ac_idx = torch.norm(ac_norms[valid_actions] - torch.from_numpy(ac_norm), dim = 1).argmin()  # First index where condition is True
                ac_idx = np.nonzero(valid_actions)[0][ac_idx] # Get index into full ac_norms
            else:
                print(f"\033[93mNo valid action for threshold {thresh:1.2e}, min value {np.max(qvals):1.2e}\033[0m")
                ac_idx = qvals.argmax()
            #if ac_idx != 0:
            #    #print('CBF filtering!')
            #elif ac_idx == -1:
            #    #print("LR filtering")
            ac_norm = ac_norms[ac_idx].cpu().unsqueeze(0).numpy()
            action['action'] = (ac_norm + 1) * 0.5 * (max_ac - min_ac) + min_ac
            
            taken_vals[episode].append(qvals[ac_idx])
        elif filter_mode == 'least_restrictive' or filter_mode == 'lr':
            if qval < thresh:
                action['action'] = ac_safe
                taken_vals[episode].append(qvals[-1])
            else:
                taken_vals[episode].append(qvals[0])
        else: 
            taken_vals[episode].append(qvals[0])
            pass # do nothing, use the original action

        ac_prev = action['action'].squeeze()

        if sampling_video:
            video.add_frame(agent._wm, agent_state[0], obs_vec, ac_norm, Q)

        taken_actions[episode].append(action['action'])

        obs_vec, reward_vec, term_vec, trunc_vec, info_vec = envs.step(action)

        done_vec = term_vec | trunc_vec
        done = done_vec.cpu().numpy()
        obs_vec['failure'] = info_vec['is_knocked_over']
        knocked_over = knocked_over or info_vec['is_knocked_over'][0]
        num_done = done.sum()
        if num_done > 0:
            # Record outcomes
            if info_vec['final_info']['is_knocked_over'][0]:
                outcomes['fail'].append(episode)
            if info_vec['final_info']['is_obj_lifted'][0]:
                outcomes['lifted'].append(episode)
            if info_vec['final_info']['is_grasped'][0]:
                outcomes['grasped'].append(episode)
            if info_vec['final_info']['success'][0] and not info_vec['final_info']['is_knocked_over'][0]:
                outcomes['success'].append(episode)
            if episode == 7:
                pass
            # Record numbers
            print(f"{episode}:\tfails = {len(outcomes['fail'])}\tsuccess = {len(outcomes['success'])}\tlifted = {len(outcomes['lifted'])}\tgrasped = {len(outcomes['grasped'])}\t\tresetting!")

            obs_vec, info = envs.reset()
            obs_vec['failure'] = info['is_knocked_over']
            done_vec = np.zeros(envs.num_envs, bool)

            plt.plot(sample_vals[episode], color="grey")
            plt.plot(safe_vals[episode], color="green", linestyle='-', marker='x')
            plt.plot(taken_vals[episode], color="black", linestyle='-', marker='x')
            plt.savefig(output_dir / f"safe_sample_vals{episode}.png")
            plt.clf()

            successes += 0 if knocked_over else 1
            knocked_over = False
            grasped = False
            lifted = False

            agent_state = None
            if policy == 'mpc':
                mpc = EndEffectorMPC(get_block_pose(envs)[0, :3], 10)
            if sampling_video:
                video.save(video_dir / (str(episode) + ".mp4"))

        episode += num_done

    # Plot difference between Q(x', pi_safe) and Q(x, u) as a function of Q(x, u)
    plt.clf()
    taken = np.concatenate(taken_vals)
    safe = np.concatenate(safe_vals)
    plt.scatter(taken[:-1], safe[1:] - taken[:-1], marker='.')
    plt.xlabel('Q(x, u)')
    plt.ylabel('Q(f(x, u), pi_safe) - Q(x, u)')
    plt.savefig(output_dir / 'actual_vs_taken.png')

    # Plot difference between Q(x, u) and max([Q(x', u) for u in samples]
    plt.clf()
    samples = np.concatenate(sample_vals)
    plt.scatter(taken[:-1], np.max(samples, axis=1)[1:] - taken[:-1], marker='.')
    plt.xlabel('Q(x, u)')
    plt.ylabel('max_u Q(x\', u) - Q(x, u)')
    plt.savefig(output_dir / 'taken_vs_samples.png')
    plt.scatter


    # Print outcomes
    print("did nothing: ", [o for o in range(num_trajs) if o not in outcomes['grasped'] and o not in outcomes['fail']])
    print("only grasped: ", [o for o in outcomes['grasped'] if o not in outcomes['lifted']])
    print("only lifted: ", [o for o in outcomes['lifted'] if o not in outcomes['success']])
    print("successes: ", outcomes['success'])
    print("fails: ", outcomes['fail'])
    print(f"{len(outcomes['success'])} success, {len(outcomes['fail'])} fails, ")

    # Plot endeffector trajectories
    ee_trajs = np.stack(ee_trajs)
    mask = np.max(abs(np.diff(ee_trajs[:, :, 0], axis = 1)), axis=1)

    print("singularities", np.argwhere(mask >= 0.05))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for traj in ee_trajs[np.where(mask < 0.05)[0], :, :]:
        x = traj[:, 0]
        y = traj[:, 1]
        z = traj[:, 2]
        ax.plot(x, y, z, alpha=0.6)

    plt.savefig(output_dir / "ee_trajectories.png")

    # Dump everything to a pickle
    envs._env.env.env.output_dir
    with open(envs._env.env.env.output_dir / "filter_results.pkl", "wb") as f:
        pickle.dump((sample_vals, safe_vals, taken_vals,
                     nominal_actions, safe_actions, sample_actions, taken_actions,
                     ee_trajs, outcomes, filter_mode, policy, thresh, cbf_gamma), f)

    print("done")

def main(args):
    if args.use_gp:
        run_name = 'FilterRolloutGP'
    else:
        run_name = 'FilterRolloutNoGP'

    # args.logdir = f"runs/{run_name}"
    # add time to run name

    run_name = f"{run_name}_{time.strftime('%Y%m%d-%H%M%S')}"

    args.logdir = f"runs/{run_name}"
    args.traindir = pathlib.Path(args.logdir) / "train_eps"
    args.evaldir = pathlib.Path(args.logdir) / "eval_eps"
    logdir = pathlib.Path(args.logdir).expanduser()
    args.traindir = args.traindir or logdir / "train_eps"
    args.evaldir = args.evaldir or logdir / "eval_eps"
    args.steps //= args.action_repeat
    args.eval_every //= args.action_repeat
    args.log_every //= args.action_repeat
    args.time_limit //= args.action_repeat

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Environment setup #######
    env_kwargs = dict(obs_mode=args.obs_mode, render_mode=args.render_mode, sim_backend="gpu", sensor_configs=dict())
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    if args.camera_width is not None:
        # this overrides every sensor used for observation generation
        env_kwargs["sensor_configs"]["width"] = args.camera_width
    if args.camera_height is not None:
        env_kwargs["sensor_configs"]["height"] = args.camera_height
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, human_render_camera_configs=dict(shader_pack="default"), **env_kwargs)
    

    # Env Wrappers
    max_episode_steps = gym_utils.find_max_episode_steps_value(envs.env) #60

    envs = DreamerWrapper(envs)
    eval_envs = DreamerWrapper(eval_envs)
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
    if args.save_train_video_freq is not None:
        save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
        envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=max_episode_steps, video_fps=30)
    eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_on_reset=True, save_trajectory=args.save_trajectory, save_video=args.capture_video, trajectory_name="trajectory", max_steps_per_video=max_episode_steps, video_fps=30)

    envs = SelectAction(envs)
    eval_envs = SelectAction(eval_envs)
    envs = UUID(envs)
    eval_envs = UUID(eval_envs)
    print(f"Max episode steps: {max_episode_steps}")
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    logger = None

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    args.traindir.mkdir(parents=True, exist_ok=True)
    args.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(args.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, args.action_repeat * step)

    print("Create envs.")
    if args.offline_traindir:
        directory = args.offline_traindir.format(**vars(args))
    else:
        directory = args.traindir
    train_eps = tools.load_episodes(directory, limit=args.dataset_size)
    if args.offline_evaldir:
        directory = args.offline_evaldir.format(**vars(args))
    else:
        directory = args.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    expert_eps = collections.OrderedDict()
    
    acts = envs.single_action_space
    acts.low = np.ones_like(acts.low) * -1
    acts.high = np.ones_like(acts.high) # need to normalize actions 
    print("Action Space", acts)
    
    args.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    train_dataset = make_dataset(train_eps, args)
    expert_dataset = make_dataset(expert_eps, args)

    agent = Dreamer(
        envs.single_observation_space,
        envs.single_action_space,
        args,
        logger,
        train_dataset,
        expert_dataset=expert_dataset,
    ).to(args.device)
    agent.requires_grad_(requires_grad=False)
    checkpoint = torch.load(args.wm_directory)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
    agent._should_pretrain._once = False
    # seed
    ac_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32) # joint action space
    ob_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,1,1536,), dtype=np.float32)

    args.state_shape = ob_space.shape or ob_space.n
    args.action_shape = ac_space.shape or ac_space.n

    args.max_action = ac_space.high[0]

    args.action_shape = ac_space.shape or ac_space.n
    args.max_action = ac_space.high[0]

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

    from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy

    print("DDPG under the Avoid annealed Bellman equation with no Disturbance has been loaded!")

    actor_net = Net(args.state_shape, hidden_sizes=args.control_net, activation=actor_activation, device=args.device)
    actor = Actor(
        actor_net, args.action_shape, max_action=args.max_action, device=args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)


    safe_policy = DDPGPolicy(
    critic,
    critic_optim,
    tau=args.tau,
    gamma=args.gamma_pyhj,
    exploration_noise=GaussianNoise(sigma=args.exploration_noise),
    reward_normalization=args.rew_norm,
    estimation_step=args.n_step,
    action_space=ac_space,
    actor=actor,
    actor_optim=actor_optim,
    actor_gradient_steps=args.actor_gradient_steps,
    ).to(args.device)
    if args.use_gp:
        filter_checkpoint = torch.load(args.filter_directory_gp)
    else:
        filter_checkpoint = torch.load(args.filter_directory_nogp)
    safe_policy.load_state_dict(filter_checkpoint)

    policy = functools.partial(agent, training=False)

    
    rollout_policy(policy, safe_policy, agent, eval_envs, num_trajs=args.num_runs, thresh=args.filter_thresh, cbf_gamma=args.cbf_gamma, filter_mode=args.filter_mode, policy=args.policy, sampling_video=args.sampling_video)
    #envs.reset()
    #print('replay')
    #replay_policy(policy, safe_policy, agent, eval_envs, '/home/kensuke/WM_CBF/ManiSkill/examples/baselines/dreamerv3-torch/runs/FilterRollout/videos/trajectory.h5')
    #print("GP metrics", agent.gp_metrics)
    #print("No GP metrics", agent.nogp_metrics)
    envs.close()
    eval_envs.close()
    


if __name__ == "__main__":
    args = tyro.cli(Args)
    loc_args = LocalArgs()

    for key, value in loc_args.__dict__.items():
        setattr(args, key, value)

    main(args)