# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# See also denoised_mdp/envs/dmc/dmc2gym/README.md for license-related
# information about these files adapted from
# https://github.com/facebookresearch/deep_bisim4control/

from typing import *

import glob
import os
import contextlib

import cv2
from gym import core, spaces
from dm_env import specs
import numpy as np
import torch

from . import local_dm_control_suite as suite
from . import natural_imgsource
from ...abc import EnvBase, AutoResetEnvBase, IndexableSized
from ...utils import as_SeedSequence, split_seed
from .... import utils


def _spec_to_box(spec, repeat=1):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.tile(np.concatenate(mins, axis=0), repeat).astype(np.float32)
    high = np.tile(np.concatenate(maxs, axis=0), repeat).astype(np.float32)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs, dtype=None):
    obs_pieces = []
    for v in obs.values():
        if np.isscalar(v):
            flat = np.array([v], dtype=dtype)
        else:
            flat = np.asarray(v.ravel(), dtype=dtype)
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCWrapper(core.Env, EnvBase):
    _bg_source: Optional[natural_imgsource.ImageSource]

    @property
    def background_source(self) -> Optional[natural_imgsource.ImageSource]:
        return self._bg_source

    @property
    def max_episode_length(self) -> int:
        return self._episode_length

    @property
    def observation_output_kind(self) -> EnvBase.ObsOutputKind:
        return self._observation_output_kind

    @property
    def action_repeat(self) -> int:
        return self._frame_skip

    def __init__(
        self,
        domain_name,
        task_name,
        resource_files,
        total_frames,
        task_kwargs={},
        visualize_reward={},
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None,
        sensor_noise_mult=0,
        observation_output_kind=EnvBase.ObsOutputKind.image_uint8,
        seed=1,
        episode_length=1000,
        spatial_jitter=0,
        spatial_jitter_randastddev=0.005,
        background_remove_mode='argmax',
    ):

        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._sensor_noise_mult = sensor_noise_mult
        self._observation_output_kind = observation_output_kind
        self._episode_length = episode_length
        self._spatial_jitter = spatial_jitter
        self._spatial_jitter_randastddev = spatial_jitter_randastddev
        self._spatial_jitter_affine_mat = np.float32([[1, 0, 0], [0, 1, 0]])
        assert background_remove_mode in {'argmax', 'dbc'}
        self._background_remove_mode = background_remove_mode


        if self._sensor_noise_mult != 0:
            assert 'noisy_sensor' not in task_kwargs
            task_kwargs = task_kwargs.copy()  # don't modify input dict
            task_kwargs.update(noisy_sensor=True)

        # create task
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs,
        )

        assert getattr(self._env.physics, 'noise_enabled', False) == (self._sensor_noise_mult != 0)

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()])
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32,
        )

        # create observation space
        self._observation_space = EnvBase.ObsOutputKind.get_observation_space(observation_output_kind, height, width)

        self._internal_state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=tuple(self._env.physics.get_state().shape),
            dtype=np.float32,
        )

        # background
        if resource_files is not None:
            shape2d = (height, width)
            files = glob.glob(os.path.expanduser(resource_files))
            assert len(files), f"Pattern {resource_files} does not match any files"
            self._bg_source = natural_imgsource.RandomVideoSource(shape2d, files, grayscale=True, total_frames=total_frames)
        else:
            self._bg_source = None

        self._steps_taken = 0
        self.seed(seed)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self):
        obs = self.render(
            height=self._height,
            width=self._width,
            camera_id=self._camera_id,
        )
        if self.background_source is not None:
            if self._background_remove_mode == 'dbc':
                mask = np.logical_and(
                    (obs[:, :, 2] > obs[:, :, 1]),
                    (obs[:, :, 2] > obs[:, :, 0]),
                )  # hardcoded for dmc. basically B is uniquely the argmax (same as dbc)
            elif self._background_remove_mode == 'argmax':
                mask = np.logical_and(
                    (obs[:, :, 2] >= obs[:, :, 1]),
                    (obs[:, :, 2] >= obs[:, :, 0]),
                )  # hardcoded for dmc. basically B in argmax. this gives better mask
            else:
                assert False, self._background_remove_mode
            bg: np.ndarray = self.background_source.get_image()
            obs[mask] = bg[mask]
        if self._spatial_jitter != 0:
            self._spatial_jitter_affine_mat[:, -1] = self._spatial_jitter_vec
            obs = cv2.warpAffine(obs, self._spatial_jitter_affine_mat, (obs.shape[1], obs.shape[0]))
        obs = self.ndarray_uint8_image_to_observation(obs, target_shape=None)

        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def internal_state_space(self):
        return self._internal_state_space

    @property
    def action_space(self):
        return self._norm_action_space

    def sample_random_action(self, size=(), np_rng=None):
        if np_rng is None:
            np_rng = np.random
        return torch.as_tensor(np_rng.uniform(-1, 1, size=tuple(size) + tuple(self.action_shape)), dtype=torch.float32)

    @property
    def dmc_env_np_random_state(self) -> np.random.RandomState:
        return cast(np.random.RandomState, self._env.task.random)

    def seed(self, seed: Union[int, np.random.SeedSequence]):
        np_random_seed, bg_seed, env_src_seed = split_seed(seed, 3)
        self._np_random = np.random.Generator(np.random.PCG64(np_random_seed))

        if self.background_source is not None:
            self.background_source.seed(bg_seed)

        int_seed = np.random.Generator(np.random.PCG64(env_src_seed)).integers(1 << 31)
        self.dmc_env_np_random_state.seed(int_seed)

    def randn(self, *size: int) -> torch.Tensor:
        return torch.as_tensor(self.randn_np(*size))

    def randn_np(self, *size: int) -> np.ndarray:
        return self._np_random.normal(size=size)

    def get_random_state(self):
        return (
            self._np_random.bit_generator.state,
            None if self.background_source is None else self.background_source.get_random_state(),
            self.dmc_env_np_random_state.get_state(),
        )

    def set_random_state(self, random_state):
        self._np_random.bit_generator.state = random_state[0]
        if self.background_source is not None:
            self.background_source.set_random_state(random_state[1])
        self.dmc_env_np_random_state.set_state(random_state[2])

    def _step_env(self, env, action):
        with contextlib.ExitStack() as stack:
            if self._sensor_noise_mult != 0:
                assert env.physics.NUM_NOISES == 1
                # background
                background = self._bg_source.get_image()
                patch_h = background.shape[0] // 4
                patch_w = background.shape[1] // 4
                patch = background[
                    (patch_h // 2) : (patch_h // 2) + patch_h,
                    (patch_w // 2) : (patch_w // 2) + patch_w,
                ]
                noise = (patch.mean() / 255 - 0.5) * self._sensor_noise_mult
                stack.enter_context(env.physics.sensor_noise([noise]))
            ts = env.step(action)
        return ts

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach()
        action = np.asarray(action, dtype=np.float32)
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        last_reward = 0
        actual_env_steps_taken = 0

        for _ in range(self._frame_skip):
            timestep = self._step_env(self._env, action)
            last_reward = timestep.reward or 0
            reward += last_reward
            done = timestep.last()
            actual_env_steps_taken += 1
            self._steps_taken += 1
            if self._steps_taken >= self._episode_length:
                done = True
            if done:
                break
            if self.background_source is not None:
                self.background_source.increment(amount=1)
            self._spatial_jitter_vec = next(self._spatial_jitter_iter)
        obs = self._get_obs()
        return obs, reward, done, EnvBase.Info(actual_env_steps_taken)  #, extra

    def reset(self) -> Tuple[torch.Tensor, EnvBase.Info]:
        self._steps_taken = 0
        timestep = self._env.reset()
        from ...utils import SmoothRandomWalker
        self._spatial_jitter_iter = iter(SmoothRandomWalker(
            d=2, dtmult=self._spatial_jitter, env=self, speed_factor=1 / 2,  # / self.action_repeat
            randastddev=self._spatial_jitter_randastddev))
        self._spatial_jitter_vec = next(self._spatial_jitter_iter)
        obs = self._get_obs()
        return obs, EnvBase.Info(0)

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)
