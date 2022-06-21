from typing import *

import sys
if sys.version_info < (3, 8):
    from typing_extensions import Protocol

import abc
import argparse
import enum

import gym
import gym.spaces
import cv2
import numpy as np
import torch


T = TypeVar('T', covariant=True)


class IndexableSized(Protocol[T]):
    def __getitem__(self, idx) -> T:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[T]:
        ...


class EnvBase(abc.ABC):
    class ObsOutputKind(enum.Enum):
        image_uint8 = argparse.Namespace(low=0, high=255, dtype=torch.uint8, np_dtype=np.uint8)
        image_float32 = argparse.Namespace(low=-0.5, high=0.5, dtype=torch.float32, np_dtype=np.float32)

        def get_observation_space(ouput_kind: 'EnvBase.ObsOutputKind', height: int, width: int, *,
                                  num_channels: int = 3) -> gym.spaces.Box:
            return gym.spaces.Box(
                low=ouput_kind.value.low,
                high=ouput_kind.value.high,
                shape=[num_channels, height, width],
                dtype=ouput_kind.value.np_dtype,
            )


    @property
    @abc.abstractmethod
    def max_episode_length(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def observation_output_kind(self) -> 'EnvBase.ObsOutputKind':
        pass

    def ndarray_uint8_image_to_observation(self, np_img: np.ndarray, *,
                                           target_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        np_chw_img = self.ndarray_uint8_image_resize(np_img, target_shape).transpose(2, 0, 1)
        if any(s < 0 for s in np_chw_img.strides) or not np_chw_img.flags['WRITEABLE']:
            # https://github.com/pytorch/pytorch/issues/64107
            np_chw_img = np_chw_img.copy()
        img = torch.as_tensor(np_chw_img)
        if self.observation_output_kind is EnvBase.ObsOutputKind.image_float32:
            img = self.uint8_image_to_float32(img)
        return img

    def process_observation_as_network_input(self, maybe_batched_observation: torch.Tensor) -> torch.Tensor:
        if self.observation_output_kind is EnvBase.ObsOutputKind.image_uint8:
            maybe_batched_observation = self.uint8_image_to_float32(maybe_batched_observation)
        assert maybe_batched_observation.dtype == torch.float32
        return maybe_batched_observation

    @staticmethod
    def uint8_image_to_float32(maybe_batched_img: torch.Tensor) -> torch.Tensor:
        # https://github.com/danijar/dreamer/blob/56d4d444dfd0582b0e79dab80aebbea74c0ce40d/dreamer.py#L333
        return maybe_batched_img.div(255).sub_(0.5)

    @staticmethod
    def ndarray_uint8_image_resize(np_img: np.ndarray, target_shape: Optional[Tuple[int, int]] = None, *,
                                   copy: bool = False) -> np.ndarray:
        assert np_img.dtype == np.uint8
        resized = False
        if target_shape is not None:
            if tuple(np_img.shape[:2]) != target_shape:
                # Resize and put channel first
                np_img = cv2.resize(np_img, target_shape[::-1], interpolation=cv2.INTER_LINEAR)
                resized = True
        if copy and not resized:
            np_img = np_img.copy()
        return np_img

    @property
    @abc.abstractmethod
    def action_repeat(self) -> int:
        pass

    observation_space: gym.spaces.Box

    @property
    def observation_shape(self) -> IndexableSized[int]:
        return self.observation_space.shape

    action_space: gym.spaces.Box

    @property
    def innermost_env_class(self) -> 'Type[EnvBase]':
        return self.__class__

    @property
    def action_shape(self) -> IndexableSized[int]:
        return self.action_space.shape

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([])

    @property
    def action_ndim(self) -> int:
        action_shape = self.action_shape
        assert len(action_shape) == 1
        return action_shape[0]

    @abc.abstractmethod
    def sample_random_action(self, size=(), np_rng=None) -> Union[float, torch.Tensor]:
        # Sample an action randomly from a uniform distribution over all valid actions
        pass

    class Info(NamedTuple):
        actual_env_steps_taken: Union[torch.Tensor, int]

    @abc.abstractmethod
    def reset(self) -> Tuple[torch.Tensor, 'EnvBase.Info']:
        pass

    @abc.abstractmethod
    def step(self, action) -> Tuple[torch.Tensor, Any, Any, 'EnvBase.Info']:
        pass

    @abc.abstractmethod
    def render(self):
        pass

    @abc.abstractmethod
    def get_random_state(self) -> Any:
        pass

    @abc.abstractmethod
    def set_random_state(self, random_state):
        pass

    @abc.abstractmethod
    def seed(self, seed: Union[int, np.random.SeedSequence]):
        pass

    def close(self):
        pass


class AutoResetEnvBase(EnvBase):
    class Info(NamedTuple):
        observation_before_reset: torch.Tensor
        actual_env_steps_taken: Union[torch.Tensor, int]

    @abc.abstractmethod
    def reset(self) -> Tuple[torch.Tensor, 'AutoResetEnvBase.Info']:
        pass

    @abc.abstractmethod
    def step(self, action) -> Tuple[torch.Tensor, Any, Any, 'AutoResetEnvBase.Info']:  # type: ignore
        pass
