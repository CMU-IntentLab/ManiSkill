import torch
import gymnasium as gym
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.types import Array
import numpy as np
import datetime
import uuid
class DreamerWrapper(gym.Wrapper):
    def __init__(self, env: BaseEnv):
        """
        Environment wrapper that repeats the action for a number of steps.
        This wrapper will perform the same action at most repeat times, if the environment is done before repeating the action repeat times, then we only return valid data (up to the done=True).

        Args:
            env (BaseEnv): The base environment to wrap.
            repeat (int): The number of times to repeat the action, repeat=1 means no action repeat (we use perform 1 action per step), repeat=2 means the action is repeated twice, so the environment will step twice with the same action.
        """
        super().__init__(env)

        spaces = env.observation_space

        pol_space_vec = gym.spaces.Box(
            low=spaces['agent']['qpos'].low,
            high=spaces['agent']['qpos'].high,
            shape=spaces['agent']['qpos'].shape,
            dtype=spaces['agent']['qpos'].dtype
        )
        pol_space = gym.spaces.Box(
            low=spaces['agent']['qpos'].low[0],
            high=spaces['agent']['qpos'].high[0],
            shape=spaces['agent']['qpos'].shape[1:],
            dtype=spaces['agent']['qpos'].dtype
        )
        img_space_vec = gym.spaces.Box(
            low=np.zeros([self.num_envs, 128, 128, 3], dtype=np.uint8),
            high=255 * np.ones([self.num_envs, 128, 128, 3], dtype=np.uint8),
            shape=[self.num_envs, 128, 128, 3],
            dtype=np.uint8
        )
        img_space = gym.spaces.Box(
            low=np.zeros([128, 128, 3], dtype=np.uint8),
            high=255 * np.ones([128, 128, 3], dtype=np.uint8),
            shape=[128, 128, 3],
            dtype=np.uint8
        )
        new_spaces = {
            "state": pol_space,
            "wrist_cam": img_space,
            "front_cam": img_space,
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        }

        new_spaces_vec = {
            "state": pol_space_vec,
            "wrist_cam": img_space_vec,
            "front_cam": img_space_vec,
            "is_first": gym.spaces.Box(0, 1, (self.num_envs,), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (self.num_envs,), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (self.num_envs,), dtype=bool),
        }

        self.observation_space = gym.spaces.Dict(new_spaces_vec)
        self.single_observation_space = gym.spaces.Dict(new_spaces)

    @property
    def num_envs(self):
        return self.base_env.num_envs

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped
   

    def step(self, action):
        final_obs, final_rew, final_terminations, final_truncations, infos = (
            super().step(action)
        )
        if type(action) is torch.Tensor:
            action[:, -1] = torch.sign(action[:, -1])  # ensure the last action is a sign
        else:
            action[:, -1] = np.sign(action[:, -1])
        done = final_terminations | final_truncations
        
        final_obs = {'state': final_obs['agent']['qpos'], # ideally EEF Pos
               'wrist_cam': final_obs['sensor_data']['hand_camera']['rgb'],
               'front_cam': final_obs['sensor_data']['base_camera']['rgb'],
               'is_first': (done*0).int(),
               'is_last': done.int(),
               'is_terminal': final_terminations.int()}

        return final_obs, final_rew, final_terminations, final_truncations, infos


    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = {'state': obs['agent']['qpos'], # ideally EEF Pos
               'wrist_cam': obs['sensor_data']['hand_camera']['rgb'],
               'front_cam': obs['sensor_data']['base_camera']['rgb'],
               'is_first': torch.tensor([1] * self.num_envs).to(obs['agent']['qpos'].device),
               'is_last': torch.tensor([0] * self.num_envs).to(obs['agent']['qpos'].device),
               'is_terminal':torch.tensor([0] * self.num_envs).to(obs['agent']['qpos'].device)}        

        return obs, info



class SelectAction(gym.Wrapper):
    def __init__(self, env: BaseEnv, key = "action"):
        """
        Environment wrapper that repeats the action for a number of steps.
        This wrapper will perform the same action at most repeat times, if the environment is done before repeating the action repeat times, then we only return valid data (up to the done=True).

        Args:
            env (BaseEnv): The base environment to wrap.
            repeat (int): The number of times to repeat the action, repeat=1 means no action repeat (we use perform 1 action per step), repeat=2 means the action is repeated twice, so the environment will step twice with the same action.
        """
        super().__init__(env)

   
        self._key = key

    def step(self, action):
        return self.env.step(action[self._key])


class UUID(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = env.num_envs
        self.env_ids = [str(uuid.uuid4()) for _ in range(self.num_envs)]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.env_ids = [str(uuid.uuid4()) for _ in range(self.num_envs)]
        info["env_ids"] = self.env_ids
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, infos = self.env.step(action)
        infos["env_ids"] = self.env_ids
        return obs, reward, terminated, truncated, infos