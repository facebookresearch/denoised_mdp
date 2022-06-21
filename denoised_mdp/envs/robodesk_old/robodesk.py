"""Desk environment with Franka Panda arm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib

from typing import *
import os

from dm_control import mujoco
from dm_control.utils import inverse_kinematics
import gym
import numpy as np
from PIL import Image


class RoboDesk(gym.Env):
    r"""Multi-task manipulation environment."""

    np_rng: np.random.Generator

    def __init__(self, task='open_slide', reward='dense', action_repeat=1,
                 episode_length=500, seed=None, image_size=64, xml_file='desk.xml',
                 camera_raw_render_size=None, resize_interpolation=Image.LANCZOS):
        assert reward in ('dense', 'sparse', 'success'), reward
        self.task = task

        model_path = os.path.join(os.path.dirname(__file__), f'assets/{xml_file}')
        self.physics = mujoco.Physics.from_xml_path(model_path)
        self.physics_copy = self.physics.copy(share_model=True)
        self.physics_copy.data.qpos[:] = self.physics.data.qpos[:]

        # Robot constants
        self.num_joints = 9
        self.joint_bounds = self.physics.model.actuator_ctrlrange.copy()

        # Environment params
        self.image_size = image_size
        self.action_dim = 5
        self.reward = reward
        self.success = None
        if camera_raw_render_size is None:
            if image_size == 64:
                assert self.optimal_min_raw_render_size == 120
            camera_raw_render_size = min(480, self.optimal_min_raw_render_size)
        self.camera_raw_render_size = camera_raw_render_size
        self.resize_interpolation = resize_interpolation

        # Action space
        self.end_effector_scale = 0.01
        self.wrist_scale = 0.02
        self.joint_scale = 0.02

        # Episode length
        self.action_repeat = action_repeat
        self.num_steps = 0
        self.episode_length = episode_length

        self.original_pos = {}
        self.previous_z_angle = None
        self.total_rotation = 0

        self.core_tasks = list(self.reward_functions)[0:12]
        self.all_tasks = list(self.reward_functions)

        self.seed(seed)

    @property
    def reward_functions(self):
        # pylint: disable=g-long-lambda
        return {
            # Core tasks
            'open_slide': self._slide_reward,
            'open_drawer': self._drawer_reward,
            'push_green': (lambda reward_type: self._button_reward(
                    'green', reward_type)),
            'stack': self._stack_reward,
            'upright_block_off_table': (lambda reward_type: self._push_off_table(
                    'upright_block', reward_type)),
            'flat_block_in_bin': (lambda reward_type: self._put_in_bin(
                    'flat_block', reward_type)),
            'flat_block_in_shelf': (lambda reward_type: self._put_in_shelf(
                    'flat_block', reward_type)),
            'lift_upright_block': (lambda reward_type: self._lift_block(
                    'upright_block', reward_type)),
            'lift_ball': (lambda reward_type: self._lift_block(
                    'ball', reward_type)),

            # Extra tasks
            'push_blue': (lambda reward_type: self._button_reward(
                    'blue', reward_type)),
            'push_red': (lambda reward_type: self._button_reward(
                    'red', reward_type)),
            'flat_block_off_table': (lambda reward_type: self._push_off_table(
                    'flat_block', reward_type)),
            'ball_off_table': (lambda reward_type: self._push_off_table(
                    'ball', reward_type)),
            'upright_block_in_bin': (lambda reward_type: self._put_in_bin(
                    'upright_block', reward_type)),
            'ball_in_bin': (lambda reward_type: self._put_in_bin(
                    'ball', reward_type)),
            'upright_block_in_shelf': (lambda reward_type: self._put_in_shelf(
                    'upright_block', reward_type)),
            'ball_in_shelf': (lambda reward_type: self._put_in_shelf(
                    'ball', reward_type)),
            'lift_flat_block': (lambda reward_type: self._lift_block(
                    'flat_block', reward_type)),
        }

    def seed(self, seed: Union[int, np.random.SeedSequence, None] = None):
        from ..utils import as_SeedSequence
        ss = as_SeedSequence(seed)
        self.np_rng = np.random.Generator(np.random.PCG64(ss))

    def get_random_state(self):
        return (
            self.np_rng.bit_generator.state,
        )

    def set_random_state(self, random_state):
        self.np_rng.bit_generator.state = random_state[0]

    @property
    def action_space(self):
        return gym.spaces.Box(-np.ones(self.action_dim, dtype=np.float32), np.ones(self.action_dim, dtype=np.float32))

    @property
    def observation_space(self):
        spaces = {
            'image': gym.spaces.Box(
                    0, 255, (self.image_size, self.image_size, 3), np.uint8),
            'qpos_robot': gym.spaces.Box(self.joint_bounds[:, 0], self.joint_bounds[:, 1]),
            'qvel_robot': gym.spaces.Box(-np.inf, np.inf, (9,), np.float32),
            'end_effector': gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
            'qpos_objects': gym.spaces.Box(-np.inf, np.inf, (26,), np.float32),
            'qvel_objects': gym.spaces.Box(-np.inf, np.inf, (26,), np.float32),
        }
        return gym.spaces.Dict(spaces)

    @property
    def cropbox(self):
        return np.array((16.75, 25.0, 105.0, 88.75)) * self.camera_raw_render_size / 120  # h/w ~= 0.72

    @property
    def optimal_min_raw_render_size(self):
        return int(np.round(self.image_size * 120 / (88.75 - 25)))

    def _create_camera(self, movable=False):
        params = {'distance': 1.8, 'azimuth': 90, 'elevation': -60, 'size': self.camera_raw_render_size}
        if movable:
            camera = mujoco.MovableCamera(
                physics=self.physics, height=params['size'],
                width=params['size'])
        else:
            camera = mujoco.Camera(
                physics=self.physics, height=params['size'],
                width=params['size'], camera_id=-1)
        camera._render_camera.distance = params['distance']  # pylint: disable=protected-access
        camera._render_camera.azimuth = params['azimuth']  # pylint: disable=protected-access
        camera._render_camera.elevation = params['elevation']  # pylint: disable=protected-access
        camera._render_camera.lookat[:] = [0, 0.535, 1.1]  # pylint: disable=protected-access
        return camera

    def render(self, mode='rgb_array', resize=True, camera=None):
        user_camera = camera is not None
        camera = camera or self._create_camera()
        image = camera.render(depth=False, segmentation=False)
        if not user_camera:
            camera._scene.free()  # pylint: disable=protected-access

        if resize is not False:
            if resize is True:
                target_size = [self.image_size, self.image_size]
            else:
                target_size = resize
            image = Image.fromarray(image).crop(box=self.cropbox)
            image = image.resize(
                target_size,
                resample=self.resize_interpolation,
            )
            image = np.asarray(image)
        return image

    def _ik(self, pos):
        from absl import logging as absl_logging

        vb = absl_logging.get_verbosity()
        absl_logging.set_verbosity('error')
        out = inverse_kinematics.qpos_from_site_pose(
            self.physics_copy, 'end_effector', pos,
            joint_names=('panda0_joint1', 'panda0_joint2', 'panda0_joint3',
                         'panda0_joint4', 'panda0_joint5', 'panda0_joint6'),
            inplace=True)
        absl_logging.set_verbosity(vb)
        return out.qpos[:]

    def _action_to_delta_joint(self, unscaled_value, joint_bounds):
        """Convert actions from [-1, 1] range to joint bounds."""
        joint_range = joint_bounds[1] - joint_bounds[0]
        return (((unscaled_value + 1) * joint_range) / 2) + joint_bounds[0]

    def _convert_action(self, full_action):
        """Converts action from [-1, 1] space to desired joint position."""
        full_action = np.asarray(full_action)

        delta_action = full_action[0:3] * self.end_effector_scale
        position = (
                self.physics.named.data.site_xpos['end_effector'] + delta_action)

        joint = self._ik(position)
        delta_wrist = self._action_to_delta_joint(full_action[3], self.joint_bounds[6])
        joint[6] = ((self.wrist_scale * delta_wrist) + self.physics.named.data.qpos[6])
        joint[6] = np.clip(joint[6], self.joint_bounds[6][0], self.joint_bounds[6][1])
        joint[7] = self._action_to_delta_joint(full_action[4], self.joint_bounds[7])
        joint[8] = joint[7]
        return joint

    def before_each_physics_step(self, action):
        pass

    def after_each_physics_step(self, action):
        pass

    def step(self, action):
        total_reward = 0
        actual_env_steps_taken = 0
        done = False

        for _ in range(self.action_repeat):
            joint_position = self._convert_action(action)
            self.before_each_physics_step(action)
            for _ in range(10):
                self.physics.data.ctrl[0:9] = joint_position[0:9]
                # Ensure gravity compensation stays enabled.
                self.physics.data.qfrc_applied[0:9] = self.physics.data.qfrc_bias[0:9]
                self.physics.step()
            self.physics_copy.data.qpos[:] = self.physics.data.qpos[:]
            self.after_each_physics_step(action)

            if self.reward == 'dense':
                total_reward += self._get_task_reward(self.task, 'dense_reward')
            elif self.reward == 'sparse':
                total_reward += float(self._get_task_reward(self.task, 'success'))
            elif self.reward == 'success':
                if self.success:
                    total_reward += 0  # Only give reward once in case episode continues.
                else:
                    self.success = self._get_task_reward(self.task, 'success')
                    total_reward += float(self.success)
            else:
                raise ValueError(self.reward)

            self.num_steps += 1
            actual_env_steps_taken += 1
            if self.episode_length is not None and self.num_steps >= self.episode_length:
                done = True
                break
        info = {'discount': 1.0, 'actual_env_steps_taken': actual_env_steps_taken}
        obs = self._get_obs()
        return obs, total_reward, done, info

    def _get_init_robot_pos(self):
        init_joint_pose = np.array(
            [-0.30, -0.4, 0.28, -2.5, 0.13, 1.87, 0.91, 0.01, 0.01])
        init_joint_pose += 0.15 * self.np_rng.uniform(
            low=self.physics.model.actuator_ctrlrange[:self.num_joints, 0],
            high=self.physics.model.actuator_ctrlrange[:self.num_joints, 1])
        return init_joint_pose

    def reset(self):
        """Resets environment."""
        self.success = False
        self.num_steps = 0

        self.physics.reset()

        # Randomize object positions.
        self.physics.named.data.qpos['drawer_joint'] -= 0.10 * self.np_rng.uniform()
        self.physics.named.data.qpos['slide_joint'] += 0.20 * self.np_rng.uniform()

        self.physics.named.data.qpos['flat_block'][0] += 0.3 * self.np_rng.uniform()
        self.physics.named.data.qpos['flat_block'][1] += 0.07 * self.np_rng.uniform()
        self.physics.named.data.qpos['ball'][0] += 0.48 * self.np_rng.uniform()
        self.physics.named.data.qpos['ball'][1] += 0.08 * self.np_rng.uniform()
        self.physics.named.data.qpos['upright_block'][0] += (
            0.3 * self.np_rng.uniform() + 0.05)
        self.physics.named.data.qpos['upright_block'][1] += (
            0.05 * self.np_rng.uniform())

        # Set robot position.
        self.physics.data.qpos[:self.num_joints] = self._get_init_robot_pos()
        self.physics.data.qvel[:self.num_joints] = np.zeros(9)

        self.reset_after_physics_before_rendering()

        # Relax object intersections.
        self.physics.forward()

        # Copy physics state into IK simulation.
        self.physics_copy.data.qpos[:] = self.physics.data.qpos[:]
        self.original_pos['ball'] = self.physics.named.data.xpos['ball']
        self.original_pos['upright_block'] = self.physics.named.data.xpos['upright_block']
        self.original_pos['flat_block'] = self.physics.named.data.xpos['flat_block']

        self.drawer_opened = False

        return self._get_obs()

    @property
    def _symbolic_state_size(self) -> int:
        return 37 * 2

    def _get_symbolic_state(self) -> np.ndarray:
        return np.concatenate(
            [
                self.physics.data.qpos[:37],
                self.physics.data.qvel[:37],
            ],
            axis=-1,
        )

    def _did_not_move(self, block_name):
        current_pos = self.physics.named.data.xpos[block_name]
        dist = np.linalg.norm(current_pos - self.original_pos[block_name])
        return dist < 0.01

    def _total_movement(self, block_name, max_dist=5.0):
        current_pos = self.physics.named.data.xpos[block_name]
        dist = np.linalg.norm(current_pos - self.original_pos[block_name])
        return dist / max_dist

    def _get_dist_reward(self, object_pos, max_dist=1.0):
        eepos = self.physics.named.data.site_xpos['end_effector']
        dist = np.linalg.norm(eepos - object_pos)
        reward = 1 - (dist / max_dist)
        return max(0, min(1, reward))

    def _slide_reward(self, reward_type='dense_reward'):
        blocks = ['flat_block', 'upright_block', 'ball']
        if reward_type == 'dense_reward':
            door_pos = self.physics.named.data.qpos['slide_joint'][0] / 0.6
            target_pos = (
                self.physics.named.data.site_xpos['slide_handle'] - np.array([0.15, 0, 0]))
            dist_reward = self._get_dist_reward(target_pos)
            did_not_move_reward = (0.33 * self._did_not_move(blocks[0]) +
                                   0.33 * self._did_not_move(blocks[1]) +
                                   0.34 * self._did_not_move(blocks[2]))
            task_reward = (0.75 * door_pos) + (0.25 * dist_reward)
            return (0.9 * task_reward) + (0.1 * did_not_move_reward)
        elif reward_type == 'success':
            return 1 * (self.physics.named.data.qpos['slide_joint'] > 0.55)

    def _drawer_reward(self, reward_type='dense_reward'):
        if reward_type == 'dense_reward':
            drawer_pos = abs(self.physics.named.data.qpos['drawer_joint'][0]) / 0.3
            dist_reward = self._get_dist_reward(
                self.physics.named.data.geom_xpos['drawer_handle'])
            return (0.75 * drawer_pos) + (0.25 * dist_reward)
        elif reward_type == 'success':
            return 1 * (self.physics.named.data.qpos['drawer_joint'] < -0.2)

    def _button_reward(self, color, reward_type='dense_reward'):
        press_button = (
            self.physics.named.data.qpos[color + '_light'][0] < -0.00453)
        if reward_type == 'dense_reward':
            dist_reward = self._get_dist_reward(
                self.physics.named.data.xpos[color + '_button'])
            return (0.25 * press_button) + (0.75 * dist_reward)
        elif reward_type == 'success':
            return 1.0 * press_button

    def _stack_reward(self, reward_type='dense_reward'):
        target_offset = [0, 0, 0.0377804]
        current_offset = (
            self.physics.named.data.xpos['upright_block'] - self.physics.named.data.xpos['flat_block'])

        offset_difference = np.linalg.norm(target_offset - current_offset)

        dist_reward = self._get_dist_reward(self.physics.named.data.xpos['upright_block'])

        if reward_type == 'dense_reward':
            return -offset_difference + dist_reward
        elif reward_type == 'success':
            return offset_difference < 0.04

    def _push_off_table(self, block_name, reward_type='dense_reward'):
        blocks = ['flat_block', 'upright_block', 'ball']
        blocks.remove(block_name)
        if reward_type == 'dense_reward':
            block_pushed = (1 - (self.physics.named.data.xpos[block_name][2] /
                                                     self.original_pos[block_name][2]))
            block_0_stay_put = (1 - self._total_movement(blocks[0]))
            block_1_stay_put = (1 - self._total_movement(blocks[1]))
            reward = ((0.8 * block_pushed) + (0.1 * block_0_stay_put) +
                                (0.1 * block_1_stay_put))
            reward = max(0, min(1, reward))
            dist_reward = self._get_dist_reward(
                    self.physics.named.data.xpos[block_name])
            return (0.75 * reward) + (0.25 * dist_reward)
        elif reward_type == 'success':
            return 1 * ((self.physics.named.data.qpos[block_name][2] < 0.6) and
                                    self._did_not_move(blocks[0]) and
                                    self._did_not_move(blocks[1]))

    def _put_in_bin(self, block_name, reward_type='dense_reward'):
        pos = self.physics.named.data.xpos[block_name]
        success = (pos[0] > 0.28) and (pos[0] < 0.52) and (pos[1] > 0.38) and (
                pos[1] < 0.62) and (pos[2] > 0) and (pos[2] < 0.4)
        if reward_type == 'dense_reward':
            dist_reward = self._get_dist_reward(
                    self.physics.named.data.xpos[block_name])
            return (0.5 * dist_reward) + (0.5 * float(success))
        elif reward_type == 'success':
            return 1 * success

    def _put_in_shelf(self, block_name, reward_type='dense_reward'):
        x_success = (self.physics.named.data.xpos[block_name][0] > 0.2)
        y_success = (self.physics.named.data.xpos[block_name][1] > 1.0)
        success = x_success and y_success
        blocks = ['flat_block', 'upright_block', 'ball']
        blocks.remove(block_name)
        if reward_type == 'dense_reward':
            target_x_y = np.array([0.4, 1.1])
            block_dist_reward = 1 - (np.linalg.norm(
                target_x_y - self.physics.named.data.xpos[block_name][0:2]))
            dist_reward = self._get_dist_reward(
                self.physics.named.data.xpos[block_name])
            block_0_stay_put = (1 - self._total_movement(blocks[0]))
            block_1_stay_put = (1 - self._total_movement(blocks[1]))
            block_in_shelf = ((0.33 * dist_reward) + (0.33 * block_dist_reward) +
                                                (0.34 * float(success)))
            reward = ((0.5 * block_in_shelf) + (0.25 * block_0_stay_put) +
                                (0.25 * block_1_stay_put))
            return reward
        elif reward_type == 'success':
            return 1 * success

    def _lift_block(self, block_name, reward_type='dense_reward'):
        if reward_type == 'dense_reward':
            dist_reward = self._get_dist_reward(
                self.physics.named.data.xpos[block_name])
            block_reward = (
                self.physics.named.data.xpos[block_name][2] - self.original_pos[block_name][2]) * 10
            block_reward = max(0, min(1, block_reward))
            return (0.85 * block_reward) + (0.15 * dist_reward)
        elif reward_type == 'success':
            success_criteria = {'upright_block': 0.86, 'ball': 0.81, 'flat_block': 0.78}
            threshold = success_criteria[block_name]
            return 1 * (self.physics.named.data.xpos[block_name][2] > threshold)

    def _get_task_reward(self, task, reward_type, clamp=True):
        reward = self.reward_functions[task](reward_type)
        if clamp:
            reward = max(0, min(1, reward))
        return reward

    def _get_obs(self):
        return {
            'image': self.render(resize=True),
            'qpos_robot': self.physics.data.qpos[:self.num_joints].copy(),
            'qvel_robot': self.physics.data.qvel[:self.num_joints].copy(),
            'end_effector': self.physics.named.data.site_xpos['end_effector'],
            'qpos_objects': self.physics.data.qvel[self.num_joints:].copy(),
            'qvel_objects': self.physics.data.qvel[self.num_joints:].copy(),
        }
