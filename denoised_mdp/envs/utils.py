# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import sys
if sys.version_info < (3, 8):
    from typing_extensions import Protocol

import os
import weakref

import gym
import gym.spaces
import numpy as np
import torch

from .. import utils

from .abc import EnvBase, AutoResetEnvBase, IndexableSized


def make_batched_auto_reset_env(env_fn: Callable[[np.random.SeedSequence], EnvBase], seed, batch_shape) -> AutoResetEnvBase:
    r"""
    Given
        env_fn: np.random.SeedSequence -> EnvBase
        seed: int | np.random.SeedSequence | None
        batch_shape: torch.Size
    Returns
        An `AutoResetEnv` (of the ones created by `env_fn`) that is batched
        with shape `batch_shape`, seeded by `seed`  (differently for each env
        in the batch).
    """
    auto_reset_env_fn = lambda seed: AutoResetWrapper(env_fn(seed))
    if len(batch_shape) > 0:
        assert len(batch_shape) == 1
        from .utils import SyncVectorEnv
        return SyncVectorEnv(auto_reset_env_fn, batch_shape[0], seed=seed)
    else:
        return auto_reset_env_fn(seed)


class AutoResetWrapper(AutoResetEnvBase):
    def __init__(self, non_auto_reset_env: EnvBase):
        assert len(non_auto_reset_env.batch_shape) == 0, "must be scalar"
        self.non_auto_reset_env = non_auto_reset_env
        self.observation_space = self.non_auto_reset_env.observation_space
        self.action_space = self.non_auto_reset_env.action_space

    @property
    def max_episode_length(self) -> int:
        return self.non_auto_reset_env.max_episode_length

    @property
    def observation_output_kind(self) -> 'EnvBase.ObsOutputKind':
        return self.non_auto_reset_env.observation_output_kind

    @property
    def action_repeat(self) -> int:
        return self.non_auto_reset_env.action_repeat

    @property
    def innermost_env_class(self) -> Type[EnvBase]:
        return self.__class__

    def sample_random_action(self, size=(), np_rng=None) -> Union[float, torch.Tensor]:
        # Sample an action randomly from a uniform distribution over all valid actions
        return self.non_auto_reset_env.sample_random_action(size, np_rng)

    def reset(self) -> Tuple[torch.Tensor, AutoResetEnvBase.Info]:
        observation, info = self.non_auto_reset_env.reset()
        auto_reset_info = AutoResetEnvBase.Info(
            observation_before_reset=torch.empty(0),
            actual_env_steps_taken=info.actual_env_steps_taken,
        )
        return observation, auto_reset_info

    def step(self, action) -> Tuple[torch.Tensor, Any, Any, AutoResetEnvBase.Info]:
        next_observation, reward, done, step_info = self.non_auto_reset_env.step(action)
        observation_before_reset = next_observation
        actual_env_steps_taken = step_info.actual_env_steps_taken
        if done:
            next_observation, reset_info = self.non_auto_reset_env.reset()  # we scalar env :)
            actual_env_steps_taken += reset_info.actual_env_steps_taken  # add 0 in practice, but conceptually cleaner :)
        auto_reset_info = AutoResetEnvBase.Info(
            observation_before_reset=observation_before_reset,
            actual_env_steps_taken=actual_env_steps_taken,
        )
        return next_observation, reward, done, auto_reset_info

    def get_random_state(self):
        return self.non_auto_reset_env.get_random_state()

    def set_random_state(self, random_state):
        return self.non_auto_reset_env.set_random_state(random_state)

    def seed(self, seed: Union[np.random.SeedSequence, int]):
        return self.non_auto_reset_env.seed(seed)

    def render(self):
        return self.non_auto_reset_env.render()

    def close(self):
        return self.non_auto_reset_env.close()


def as_SeedSequence(seed: Union[np.random.SeedSequence, int, None]) -> np.random.SeedSequence:
    if isinstance(seed, int) or seed is None:
        seed = np.random.SeedSequence(seed)
    return seed


def split_seed(seed: Union[np.random.SeedSequence, int, None], n) -> List[np.random.SeedSequence]:
    return as_SeedSequence(seed).spawn(n)


def stack_dim0_or_as_tensor(l, dtype=None):
    # https://github.com/pytorch/pytorch/issues/26354
    if all(isinstance(t, torch.Tensor) for t in l):
        y = torch.stack(l, dim=0)
    else:
        y = torch.as_tensor(l, dtype=dtype)
    if dtype is not None:
        y = y.to(dtype)
    return y


class ConcatEnv(AutoResetEnvBase):
    envs: List[AutoResetEnvBase]

    def __init__(self, env_fns: List[Callable[[np.random.SeedSequence], AutoResetEnvBase]],
                 seed: Union[int, np.random.SeedSequence, None], share_seed=False):
        self.n = len(env_fns)
        self.share_seed = share_seed
        if share_seed:
            seeds = split_seed(seed, self.n)
        else:
            seeds = [seed for _ in env_fns]
        self.envs = [env_fn(seed=s) for env_fn, s in zip(env_fns, seeds)]
        for e in self.envs[1:]:
            assert e.batch_shape == self.envs[0].batch_shape
            assert e.action_space == self.envs[0].action_space
            assert e.observation_space == self.envs[0].observation_space

    # Resets every environment and returns observation
    def reset(self) -> Tuple[torch.Tensor, AutoResetEnvBase.Info]:
        observations = []
        infos = []
        for env in self.envs:
            o, i = env.reset()
            observations.append(o)
            infos.append(i)
        return torch.stack(observations, dim=0), self._stack_infos(infos)

    @staticmethod
    def _stack_infos(infos: List[AutoResetEnvBase.Info]) -> AutoResetEnvBase.Info:
        if len(infos):
            return AutoResetEnvBase.Info(
                observation_before_reset=infos[0].observation_before_reset.unsqueeze(0),
                actual_env_steps_taken=torch.as_tensor(infos[0].actual_env_steps_taken, dtype=torch.int64).unsqueeze(0),
            )
        else:
            return AutoResetEnvBase.Info(
                observation_before_reset=torch.stack(
                    [info.observation_before_reset for info in infos],
                    dim=0,
                ),
                actual_env_steps_taken=stack_dim0_or_as_tensor(
                    [info.actual_env_steps_taken for info in infos],
                    dtype=torch.int64,
                ),
            )

    # Steps/resets every environment and returns (observation, reward, done)
    def step(self, actions: torch.Tensor):
        observations, rewards, dones, infos = zip(*[env.step(a)for env, a in zip(self.envs, actions.unbind(0))])
        observations = list(observations)
        rewards = list(rewards)
        dones = list(dones)

        if self.n == 1:
            # unsqueeze
            observations, rewards, dones = (
                observations[0].unsqueeze(0),
                stack_dim0_or_as_tensor(rewards, dtype=torch.float32),
                stack_dim0_or_as_tensor(dones, dtype=torch.bool),
            )
        else:
            observations, rewards, dones = (
                torch.stack(observations, dim=0),
                stack_dim0_or_as_tensor(rewards, dtype=torch.float32),
                stack_dim0_or_as_tensor(dones, dtype=torch.bool),
            )

        info = self._stack_infos(list(infos))
        return observations, rewards, dones, info

    def close(self):
        [env.close() for env in self.envs]

    def seed(self, seed):
        if self.share_seed:
            seeds = split_seed(seed, self.n)
        else:
            seeds = [seed for _ in range(self.n)]
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

    def set_random_state(self, random_state):
        for env, rs in zip(self.envs, random_state):
            env.set_random_state(rs)

    def get_random_state(self):
        return [env.get_random_state() for env in self.envs]

    @property
    def innermost_env_class(self) -> Type[EnvBase]:
        return self.envs[0].innermost_env_class

    @property
    def max_episode_length(self) -> int:
        return max(e.max_episode_length for e in self.envs)

    @property
    def observation_output_kind(self) -> 'EnvBase.ObsOutputKind':
        return self.envs[0].observation_output_kind

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.envs[0].action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.envs[0].observation_space

    @property
    def action_repeat(self) -> int:
        return self.envs[0].action_repeat

    @property
    def observation_shape(self) -> IndexableSized[int]:
        return self.envs[0].observation_shape

    @property
    def action_shape(self) -> IndexableSized[int]:
        return self.envs[0].action_shape

    @utils.lazy_property
    def batch_shape(self) -> torch.Size:
        return torch.Size([self.n, *self.envs[0].batch_shape])

    def sample_random_action(self, size=(), np_rng=None) -> Union[float, torch.Tensor]:
        # Sample an action randomly from a uniform distribution over all valid actions
        if np_rng is None:
            np_rng = np.random
        rand = np_rng.uniform(-1, 1, size=tuple(size) + tuple(self.action_shape))
        if len(size) > 0:
            return torch.as_tensor(rand, dtype=torch.float32)
        else:
            return rand

    def render(self):
        raise RuntimeError()


class SyncVectorEnv(ConcatEnv):
    def __init__(self, env_fn: Callable[[np.random.SeedSequence], AutoResetEnvBase], n, seed):
        super().__init__(
            [env_fn for _ in range(n)],
            seed,
            share_seed=False,
        )


class RandnEnvProtocol(Protocol):
    def randn(self, *size: int) -> torch.Tensor: ...

RandnEnvT = TypeVar('RandnEnvT', bound=RandnEnvProtocol)


class SmoothRandomWalker(Generic[RandnEnvT]):
    def __init__(self, d=3, pull=0.1, pullr=0.075, boundaryr=np.inf,
                 randastddev=0.005, vdecay=0.9, dtmult=1, speed_factor=1,
                 *, env: RandnEnvProtocol):
        self.d = d
        self.pull = pull
        self.pullr = pullr
        self.boundaryr = boundaryr
        self.randastddev = randastddev
        self.vdecay = vdecay
        self.dtmult = dtmult
        self.speed_factor = speed_factor
        self.env = weakref.ref(env)

    def __iter__(self):
        if not isinstance(self.dtmult, torch.Tensor) and self.dtmult == 0:
            while True:
                yield 0
        else:
            dt = self.env().randn(self.d).mul_(self.pullr / np.sqrt(self.d))
            v = self.env().randn(self.d).mul_(self.randastddev)
            while True:
                dtnorm = np.linalg.norm(dt)
                if dtnorm > self.boundaryr:
                    dt = dt / dtnorm * self.boundaryr
                yield dt * self.dtmult
                a = torch.randn(self.d).mul_(self.randastddev)
                if dtnorm >= self.pullr:
                    a = -self.pull * dt
                v += a * self.speed_factor
                v = v * (self.vdecay ** self.speed_factor)
                dt = dt + v * self.speed_factor


def get_kinetics_dir():
    KINETICS_DIR = os.environ.get('KINETICS_DIR', default=os.path.expanduser('~/kinetics/070618/400'))
    if not os.path.isdir(KINETICS_DIR):
        raise RuntimeError(
            'Cannot find Kinetics dataset, either specify the environment flag '
            'KINETICS_DIR or place it at ~/kinetics/070618/400/'
        )
    return KINETICS_DIR
