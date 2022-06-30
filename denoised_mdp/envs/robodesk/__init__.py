# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import os
import contextlib

import torch
import numpy as np
import gym.spaces

from .robodesk.robodesk import RoboDeskWithTV as OriginalRoboDeskWithTV

from ..abc import EnvBase, AutoResetEnvBase


# Inverse Kinetics in robodesk.step() can fail to converge in early training when action is bad.
# It throws an annoying warning message if the code uses `logging`, which we do.
# This suppresses that message.
@contextlib.contextmanager
def absl_no_warning_log():
    from absl import logging as absl_logging
    vb = absl_logging.get_verbosity()
    absl_logging.set_verbosity('error')
    yield
    absl_logging.set_verbosity(vb)


class RoboDeskEnv(EnvBase):
    inner: OriginalRoboDeskWithTV

    def randn(self, *size):
        return torch.as_tensor(self.np_rng.normal(size=size), dtype=torch.float32)

    def rand(self, *size):
        return torch.as_tensor(self.np_rng.uniform(size=size), dtype=torch.float32)

    def __init__(self, *, task='open_slide', reward='dense', action_repeat=1,
                 episode_length=500, image_size=96, distractors='all', tv_video_file_pattern=None,
                 observation_output_kind: EnvBase.ObsOutputKind):
        self.inner = OriginalRoboDeskWithTV(task, reward, action_repeat, episode_length, image_size,
                                            distractors, tv_video_file_pattern)
        # Require below so that total #steps is the same regardless of AR,
        # and that the actual #steps taken each step is fixed and known.
        assert self.inner.episode_length % self.inner.action_repeat == 0
        self._observation_output_kind = observation_output_kind
        self.observation_space = EnvBase.ObsOutputKind.get_observation_space(observation_output_kind, image_size, image_size)
        self.action_space = self.inner.action_space

    @property
    def max_episode_length(self) -> int:
        return self.inner.episode_length

    @property
    def observation_output_kind(self) -> 'EnvBase.ObsOutputKind':
        return self._observation_output_kind

    @property
    def action_repeat(self) -> int:
        return self.inner.action_repeat

    def sample_random_action(self, size=(), np_rng=None) -> Union[float, torch.Tensor]:
        # Sample an action randomly from a uniform distribution over all valid actions
        if np_rng is None:
            np_rng = np.random
        return torch.as_tensor(np_rng.uniform(-1, 1, size=tuple(size) + tuple(self.action_shape)), dtype=torch.float32)

    def reset(self) -> Tuple[torch.Tensor, EnvBase.Info]:
        obs = self.inner.reset()
        obs = self.ndarray_uint8_image_to_observation(np.asarray(obs['image']))
        return obs, EnvBase.Info(actual_env_steps_taken=0)

    def step(self, action) -> Tuple[torch.Tensor, float, bool, EnvBase.Info]:
        if isinstance(action, torch.Tensor):
            action = action.detach()
        with absl_no_warning_log():
            obs, reward, done, _ = self.inner.step(np.asarray(action))
        obs = self.ndarray_uint8_image_to_observation(np.asarray(obs['image']))
        return obs, reward, done, EnvBase.Info(actual_env_steps_taken=self.action_repeat)

    def render(self):
        return self.inner.render()

    def get_random_state(self) -> Any:
        return self.inner.get_random_state()

    def set_random_state(self, random_state):
        return self.inner.set_random_state(random_state)

    def seed(self, seed: Union[int, np.random.SeedSequence]):
        return self.inner.seed(seed)


def _make_one_env(spec: str, observation_output_kind: AutoResetEnvBase.ObsOutputKind,
                  seed: np.random.SeedSequence, max_episode_length: int, action_repeat: int) -> RoboDeskEnv:
    from ..utils import get_kinetics_dir  # avoid circular imports
    task, noise_spec = spec.rsplit('_', 1)
    if noise_spec == 'noisy':
        env = RoboDeskEnv(task=task, episode_length=max_episode_length, action_repeat=action_repeat,
                          observation_output_kind=observation_output_kind, distractors='all',
                          tv_video_file_pattern=os.path.join(get_kinetics_dir(), 'train/driving_car/*.mp4'))
    elif noise_spec == 'noiseless':
        env = RoboDeskEnv(task=task, episode_length=max_episode_length, action_repeat=action_repeat,
                          observation_output_kind=observation_output_kind, distractors='none')
    else:
        raise ValueError(f'Unexpected noise_spec: {noise_spec}')
    env.seed(seed)
    return env


def make_env(spec: str, observation_output_kind: AutoResetEnvBase.ObsOutputKind,
             seed: Union[int, np.random.SeedSequence, None], max_episode_length: int,
             action_repeat: int, batch_shape: Tuple[int, ...] = ()) -> AutoResetEnvBase:
    from ..utils import make_batched_auto_reset_env  # avoid circular imports
    return make_batched_auto_reset_env(
        lambda seed: _make_one_env(spec, observation_output_kind, seed, max_episode_length, action_repeat),
        seed, batch_shape)
