# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import sys
if sys.version_info < (3, 8):
    from typing_extensions import Protocol

import abc
import dataclasses

import torch
import torch.nn as nn

from .networks import (
    TransitionModel, ActorModel, TransitionOutputWithPosterior, TransitionLatentState,
)

if TYPE_CHECKING:
    from ..memory import ExperienceReplay


class LatentStateProtocol(Protocol):
    def detach(self) -> 'LatentStateProtocol': ...


LatentStateT = TypeVar('LatentStateT', bound=LatentStateProtocol)


class AgentBase(nn.Module, Generic[LatentStateT], metaclass=abc.ABCMeta):
    actor_model: ActorModel
    transition_model: TransitionModel

    @property
    def device(self) -> torch.device:
        return self.actor_model.net.module[-1].weight.device

    @abc.abstractmethod
    def init_latent_state(self, *, batch_shape: Tuple[int, ...] = ()) -> LatentStateT:
        # The first latent state.
        pass

    @abc.abstractmethod
    def action_from_init_latent_state(self, *, batch_shape: Tuple[int, ...] = (), device=None) -> Any:
        # The first action. Can be whatever fixed tensor, e.g., all zeros following Dreamer.
        pass

    @abc.abstractmethod
    def model_learning_parameters(self) -> Iterable[torch.nn.Parameter]:
        pass

    @dataclasses.dataclass(frozen=True)
    class TrainOutput:
        transition_output: TransitionOutputWithPosterior
        posterior_latent_state: LatentStateT
        observation_prediction: torch.distributions.Normal
        reward_prediction: torch.distributions.Normal
        reward_denoised_prediction_mean: torch.Tensor

    @abc.abstractmethod
    def train_reconstruct(self, data: 'ExperienceReplay.Data') -> TrainOutput:
        pass

    class ImagineOutput(NamedTuple):
        latent_states: TransitionLatentState  # [T x B]
        reward_mean: torch.Tensor

    @abc.abstractmethod
    def imagine_ahead_noiseless(
        self,
        previous_latent_state: LatentStateT,
        planning_horizon: int = 12, freeze_latent_model: bool = True,
    ) -> ImagineOutput:
        pass

    @abc.abstractmethod
    def posterior_rsample_one_step(self, latent_state: LatentStateT,  # h/s_{t-1}
                                   action: torch.Tensor,  # a_{t-1}
                                   next_observation: torch.Tensor,  # o_{t}
                                   reward: torch.Tensor,  # r_{t}
                                   next_observation_nonfirststep: Optional[torch.Tensor] = None,
                                   ) -> LatentStateT:
        pass

    @abc.abstractmethod
    def convert_latent_state_to_actor_input(self, latent_state: LatentStateT) -> Union[TransitionLatentState, torch.Tensor]:
        pass

    def act(self, latent_state: LatentStateT, *,
            explore: bool = False, action_noise_stddev=0.3,  # whether to add noise
            ) -> torch.Tensor:
        action: torch.Tensor = self.actor_model.get_action(
            self.convert_latent_state_to_actor_input(latent_state),
            det=not (explore),
        )
        if explore and action_noise_stddev > 0:
            # Add gaussian exploration noise on top of the sampled action
            action = torch.distributions.Normal(action, action_noise_stddev).rsample()
            action = action.clamp(-1, 1)
        return action
