# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.dreamer_wrapper import DreamerWrapper, SelectAction, UUID
from gymnasium import spaces

import pathlib
import tools
import collections  
import models
import functools
import pickle

from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.exploration import GaussianNoise
from PyHJ.data import Batch
from tools import fill_expert_dataset

from config import Args
from local_config import LocalArgs
import wandb

to_np = lambda x: x.detach().cpu().numpy()

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
    def __init__(self, log_wandb=False, tensorboard = None) -> None:
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

        do_sample = True
        if self._args.eval_state_mean:
            do_sample = False
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"], sample=do_sample)
        feat = self._wm.dynamics.get_feat(latent)

        gp = self._wm.heads["margin_gp"](feat)[0].item()
        no_gp = self._wm.heads["margin_nogp"](feat)[0].item()

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


def traj_stats(
        nom_policy,
        safe_policy,
        agent,
        envs,
        offline_data
        ):
    
    torch.cuda.empty_cache()
    
    traj_counter = 0
    V_total = []
    Lz_noGP_total = []
    gt_failure_total = []
    success_total = []

    for ep in offline_data:
        print('\nEpisode:', traj_counter)
        traj_counter += 1
        done_vec = np.zeros(envs.num_envs, bool)
        agent_state = None

        ep_tstep = 0
        V_episode = []
        Lz_noGP_episode = []
        gt_failure_episode = []
        success_episode = []

        for ep_step in ep:
            print(f'\rTimestep:{ep_tstep}', end='', flush=True)
            obs_vec = ep_step
            _, agent_state = nom_policy(obs_vec, done_vec, agent_state)
            feat = agent._wm.dynamics.get_feat(agent_state[0]).cpu().detach().numpy()
            l_gp = torch.tanh(agent._wm.heads['margin_gp'](agent._wm.dynamics.get_feat(agent_state[0])))[0,0]
            # l_nogp = torch.tanh(agent._wm.heads['margin_nogp'](agent._wm.dynamics.get_feat(agent_state[0])))

            val = V(feat, safe_policy)[0] # this is just to check the shape of feat
            gt_failure = obs_vec['failure'][0]
            ep_tstep += 1

            V_episode.append(val)
            Lz_noGP_episode.append(l_gp)
            gt_failure_episode.append(gt_failure)
            # success_episode.append(obs_vec['success'][0])

        V_total.append(V_episode.copy())
        Lz_noGP_total.append(Lz_noGP_episode.copy())
        gt_failure_total.append(gt_failure_episode.copy())
        # success_total.append(success_episode.copy())

    return V_total, Lz_noGP_total, gt_failure_total, success_total

def main(args, offline_trajs):
    if args.use_gp:
        run_name = 'TrajStat'
    else:
        run_name = 'TrajStat'

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
    V_total, Lz_noGP_total, gt_failure_total, success_total = traj_stats(policy, safe_policy, agent, eval_envs, offline_trajs)
    envs.close()
    eval_envs.close()

    return V_total, Lz_noGP_total, gt_failure_total, success_total


def sample_episodes(episodes):
    """Iterates through episodes and yields each one."""
    for episode in episodes.values():
        yield episode

def from_generator(generator):
    """
    Takes a generator that yields episodes (dict of lists) and transforms each
    episode into a list of dictionaries, where each dictionary represents a timestep.
    """
    for episode in generator:
        keys = list(episode.keys())
        if not keys:
            continue

        # Assuming all lists in the episode dict have the same length.
        num_timesteps = len(episode[keys[0]])
        if num_timesteps == 0:
            continue

        # Convert the dictionary of lists to a list of dictionaries.
        episode_timesteps = [
            {key: torch.tensor(episode[key][i]).unsqueeze(0) for key in keys} for i in range(num_timesteps)
        ]
        yield episode_timesteps


def make_offline_dataset(episodes):
    generator = sample_episodes(episodes)
    dataset = from_generator(generator)
    return dataset


if __name__ == "__main__":
    offline_trajs = collections.OrderedDict()
    args = Args()
    loc_args = LocalArgs()
    for key, value in loc_args.__dict__.items():
        setattr(args, key, value)
    args.use_gp = True

    fill_expert_dataset(args, offline_trajs, add_block_failures=True)
    traj_dataset = make_offline_dataset(offline_trajs)

    V_total, Lz_noGP_total, gt_failure_total, success_total = main(args, traj_dataset)
    offline_data_dir = '/'.join((args.offline_data_path).split('/')[:-1]) + '/'

    with open(offline_data_dir + "traj_stats_eval_new.pkl", "wb") as f:
        pickle.dump((V_total, Lz_noGP_total, gt_failure_total, success_total), f)

