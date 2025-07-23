from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, PandaWristCam, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig, SceneConfig, DefaultMaterialsConfig
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

from dataclasses import asdict, dataclass, field
from mani_skill.utils.geometry import rotation_conversions

PICK_CUBE_DOC_STRING = """**Task Description:**
A simple task where the objective is to grasp a red cube with the {robot_id} robot and move it to a target goal position. This is also the *baseline* task to test whether a robot with manipulation
capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

**Randomizations:**
- the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the cube's z-axis rotation is randomized to a random angle
- the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

**Success Conditions:**
- the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
- the robot is static (q velocity < 0.2)
"""


@register_env("BlockTopple-v0", max_episode_steps=60)
class BlockToppleEnv(BaseEnv):

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda_wristcam",
        "fetch",
        "xarm6_robotiq",
        "so100",
        "widowxai",
    ]
    agent: Union[PandaWristCam, Fetch, XArm6Robotiq, SO100]

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
    
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()


        
        self.width = 0.05
        self.height = 0.2
        self.block1 = actors.build_box(
            self.scene,
            half_sizes=[self.width/2 ,self.width/2,self.height/2],
            color=[1, 0, 0, 1],
            name="block1",
            initial_pose=sapien.Pose(p=[0, 0, 0.2/2]),
        )
        self.block2 = actors.build_box(
            self.scene,
            half_sizes=[self.width/2 ,self.width/2,self.height/2],
            color=[1, 0, 0, 1],
            name="block2",
            initial_pose=sapien.Pose(p=[0, 0, 0.2/2]),
        )
        self.block3 = actors.build_box(
            self.scene,
            half_sizes=[self.width/2 ,self.width/2,self.height/2],
            color=[0, 1, 0, 1],
            name="block3",
            initial_pose=sapien.Pose(p=[0, 0, 0.2/2]),
        )
        
        

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * 0.03 * 2
                - 0.03
            )
            xyz[:, 0] += 0.15 #self.cube_spawn_center[0]
            #xyz[:, 1] += 0.025
            xyz[:, 2] = self.height / 2
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=True)
            self.block3.set_pose(Pose.create_from_pq(xyz, qs))
            xyz[:, 1] -= 0.07
            self.block1.set_pose(Pose.create_from_pq(xyz, qs))
            xyz[:, 1] += 0.07*2
            self.block2.set_pose(Pose.create_from_pq(xyz, qs))

            qpos = torch.tensor([ 0. ,0.39269908 , 0., -1.96349541, 0.,   2.3561944, 0.78539816, 0, 0.07])
            self.agent.robot.set_qpos(qpos)

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.block3.pose.raw_pose,
                tcp_to_obj_pos=self.block3.pose.p - self.agent.tcp_pose.p,
            )
        return obs

    def object_failures(self,
        angle_thresh: float = 1.0,
    ) -> torch.Tensor:
        """Check if either object is tilted (rotated too far from upright)."""
        # Get world-frame quaternions (shape: (N, 4))
        left_quat = self.block1.pose.q
        right_quat = self.block2.pose.q

        # Convert to rotation matrices and then to Euler angles (XYZ convention)
        left_mat = rotation_conversions.quaternion_to_matrix(left_quat)
        right_mat = rotation_conversions.quaternion_to_matrix(right_quat)
        left_euler = rotation_conversions.matrix_to_euler_angles(left_mat, convention="XYZ")
        right_euler = rotation_conversions.matrix_to_euler_angles(right_mat, convention="XYZ")

        # Check if X or Y angles exceed threshold
        left_fail = torch.logical_or(left_euler[:, 0].abs() > angle_thresh,
                    (left_euler[:, 1].abs() > angle_thresh)).float()

        right_fail = torch.logical_or(right_euler[:, 0].abs() > angle_thresh,
                    right_euler[:, 1].abs() > angle_thresh).float()

        fail = torch.logical_or(left_fail, right_fail).float()
        return fail
    
    def object_upright_penalty(self, scale: float = 0.005, threshold: float = 0.1) -> torch.Tensor:
        """
        Returns a smooth uprightness penalty for early tilt detection without dominating loss.

        Applies only above a threshold, saturates at full tilt.
        Max unscaled penalty per step ≈ 1.0 (for full 90° tilt on both axes),
        so recommended scale is ~0.001–0.005 depending on desired shaping strength.
        """
        left_quat = self.block1.pose.q
        right_quat = self.block2.pose.q

        left_euler = rotation_conversions.matrix_to_euler_angles(
            rotation_conversions.quaternion_to_matrix(left_quat), convention="XYZ"
        )
        right_euler = rotation_conversions.matrix_to_euler_angles(
            rotation_conversions.quaternion_to_matrix(right_quat), convention="XYZ"
        )

        left_dev = left_euler[:, :2]
        right_dev = right_euler[:, :2]

        # Subtract threshold, zero-out small tilts
        left_excess = torch.clamp(left_dev.abs() - threshold, min=0.0)
        right_excess = torch.clamp(right_dev.abs() - threshold, min=0.0)

        # Compute smooth squared penalty
        left_penalty = torch.norm(left_excess, dim=-1)
        right_penalty = torch.norm(right_excess, dim=-1)
        penalty = left_penalty + right_penalty

        # Normalize so max penalty = 1.0 (when both blocks fully flat in both axes)
        max_norm = 2 * np.sqrt(2) * (np.pi/2 - threshold)
        normalized = penalty / max_norm
        normalized = torch.clamp(normalized, max=1.0)
        return normalized * scale

    def evaluate(self):
        # object lifted above a threshold
        is_obj_lifted = (
            self.block3.pose.p[:, 2] >= self.height + self.height/2
        )
        is_grasped = self.agent.is_grasping(self.block3)
        is_robot_static = self.agent.is_static(0.2)

        is_knocked_over = self.object_failures()
        return {
            "success": is_obj_lifted & is_robot_static,
            "is_obj_lifted": is_obj_lifted,
            "is_knocked_over": is_knocked_over,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Adjust block grasp location
        block_grasp_loc = self.block3.pose.p.clone()
        block_grasp_loc[:, 2] += 0.07

        # Reaching reward
        tcp_to_obj_dist = torch.linalg.norm(
            block_grasp_loc - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        # Grasping reward
        is_grasped = info["is_grasped"]
        reward += is_grasped

        # Lifting reward (only when grasped)
        obj_to_goal_dist = torch.clamp((self.height + 0.05) - self.block3.pose.p[:, 2], min=0.0)
        dist_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += dist_reward * is_grasped 

        # Joint velocity penalty
        qvel = self.agent.robot.get_qvel()
        if self.robot_uids in ["panda_wristcam", "widowxai"]:
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        # Static reward when object is lifted
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_lifted"]

        # Joint velocity regularization
        joint_vel_pen = torch.linalg.norm(qvel, axis=1)  * 0.05
        reward -= joint_vel_pen
        # Penalty for knocking over the block
        reward -= info["is_knocked_over"]*0.5
        # action magnitude reward
        ac_rew = torch.linalg.norm(action[:, :6], axis=1) * 0.5
        reward -= ac_rew
        upright_penalty = self.object_upright_penalty(scale=0.01)
        print(upright_penalty.mean())
        reward -= upright_penalty

        #print(info["success"]) # bool
        #print(info["is_knocked_over"]) # 0. or 1. 
        #print(reward.shape, reward.dtype)
        success_mask = info["success"]
        reward[success_mask] = 4.0 - info["is_knocked_over"][success_mask]*0.5 - joint_vel_pen[success_mask] - ac_rew[success_mask]
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4

