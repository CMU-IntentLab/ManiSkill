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

to_np = lambda x: x.detach().cpu().numpy()

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
    control_mode: Optional[str] = "pd_ee_delta_pose"
    """the control mode to use for the environment"""
    render_mode: str = "all"
    """the environment rendering mode"""

    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
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
    precision: int = 32
    action_repeat: int = 1
    steps = int = 10_000_000

    eval_every: int = 10_000
    log_every: int = 10_000
    #time_limit: int = 1e3
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
    eval_state_mean: bool = False

    gamma_lx: float = 0.75
    # offline_data_path: str = '/home/kensuke/ManiSkill/examples/baselines/ppo/runs/BlockTopple-v0__ppo_rgb__1__1753308792/test_videos/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5'
    # offline_data_path: str = '/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/examples/baselines/ppo/runs/BlockTopple-v0__ppo_rgb__1__1756574937/test_videos/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5'
    offline_data_path: str = '/home/clown2/Desktop/Work/Research/ManiSkill/Maniskill/examples/baselines/dreamerv3-torch/runs/FilterRolloutGP_20250831-214144/videos/trajectory_mixed.h5'
    pretrain: int = 500
    hybrid_steps: int = 300_000
    hybrid: bool = True
from typing import Dict, Any, Union

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
    def __call__(self, obs, reset, state=None, training=True):
        #t0 = time.time()

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
        #t_train_end = time.time()

        #t_policy_start = time.time()
        policy_output, state = self._policy(obs, state, training)
        #t_policy_end = time.time()

        if training:
            self._step += len(reset)
            self._logger.step = self._args.action_repeat * self._step

        #t1 = time.time()

        #print(f"Training time: {t_train_end - t_train_start:.4f} seconds")
        #print(f"Policy time: {t_policy_end - t_policy_start:.4f} seconds")
        #print(f"Other overhead: {t1 - t0 - (t_train_end - t_train_start) - (t_policy_end - t_policy_start):.4f} seconds")
        #print(f"Total Dreamer call time: {t1 - t0:.4f} seconds")
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._args.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
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

if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

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
    args.encoder = {'mlp_keys': 'state', 'cnn_keys': '.*\_cam$', 'act': 'SiLU', 'norm': True, 'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5, 'mlp_units': 1024, 'symlog_inputs': True}
    args.decoder = {'mlp_keys': 'state', 'cnn_keys': '.*\_cam$', 'act': 'SiLU', 'norm': True, 'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5, 'mlp_units': 1024, 'cnn_sigmoid': False, 'image_dist': 'mse', 'vector_dist': 'symlog_mse', 'outscale': 1.0}
    args.actor = {'layers': 2, 'dist': 'normal', 'entropy': 3e-4, 'unimix_ratio': 0.01, 'std': 'learned', 'min_std': 0.1, 'max_std': 1.0, 'temp': 0.1, 'lr': 3e-5, 'eps': 1e-5, 'grad_clip': 100.0, 'outscale': 1.0}
    args.critic = {'layers': 2, 'dist': 'symlog_disc', 'slow_target': True, 'slow_target_update': 1, 'slow_target_fraction': 0.02, 'lr': 3e-5, 'eps': 1e-5, 'grad_clip': 100.0, 'outscale': 0.0}
    args.reward_head =  {'layers': 2, 'dist': 'symlog_disc', 'loss_scale': 1.0, 'outscale': 0.0}
    args.cont_head =  {'layers': 2, 'loss_scale': 1.0, 'outscale': 1.0}
    args.margin_head =  {'layers': 2, 'loss_scale': 1.0}
    args.grad_heads = ['decoder', 'reward', 'cont']

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
    
    envs = DreamerWrapper(envs)
    eval_envs = DreamerWrapper(eval_envs)
    envs = SelectAction(envs)
    eval_envs = SelectAction(eval_envs)
    envs = UUID(envs)
    eval_envs = UUID(eval_envs)
    
    
    # rgbd obs mode returns a dict of data, we flatten it so there is just a rgbd key and state key
    #envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=args.include_state)
    max_episode_steps = gym_utils.find_max_episode_steps_value(envs.env) #60
    print(f"Max episode steps: {max_episode_steps}")
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=max_episode_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory, save_video=args.capture_video, trajectory_name="trajectory", max_steps_per_video=max_episode_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            args = vars(args)
            args["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            args["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=False)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                args=args,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["sac", "walltime_efficient"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")


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
    if args.hybrid:
        tools.fill_expert_dataset(args, expert_eps)

    acts = envs.single_action_space
    acts.low = np.ones_like(acts.low) * -1
    acts.high = np.ones_like(acts.high) # need to normalize actions 
    print("Action Space", acts)
    
    args.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None


    if not args.offline_traindir:
        prefill = max(0, args.prefill - count_steps(args.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(args.num_actions).repeat(args.num_envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(args.num_envs, 1),
                    torch.tensor(acts.high).repeat(args.num_envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            envs,
            train_eps,
            args.traindir,
            logger,
            limit=args.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * args.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    
    

    train_dataset = make_dataset(train_eps, args)
    eval_dataset = make_dataset(eval_eps, args)
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
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # make sure eval will be executed once after args.steps
    while agent._step < args.steps + args.eval_every:
        logger.write()
        if args.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                args.evaldir,
                logger,
                is_eval=True,
                episodes=args.eval_episode_num,
            )
            if args.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        state = tools.simulate(
            agent,
            envs,
            train_eps,
            args.traindir,
            logger,
            limit=args.dataset_size,
            steps=args.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass
