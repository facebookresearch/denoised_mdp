# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import sys
if sys.version_info < (3, 8):
    from typing_extensions import Protocol

import contextlib
import attrs

import torch

from .networks import (
    ActivationKind,
    ObservationModel,
    RewardModel,
    TransitionModel,
    TransitionLatentState,
    TransitionOutputWithPosterior,
    ActorModel,
    EncoderModel,
)
from .utils import FreezeParameters
from .base import AgentBase

if TYPE_CHECKING:
    from ..memory import ExperienceReplay
    from ..envs import AutoResetEnvBase


@attrs.define(kw_only=True, auto_attribs=True)
class TransitionPartSizeSpec:
    belief_size: int = attrs.field(validator=attrs.validators.ge(0))
    state_size: int = attrs.field(validator=attrs.validators.ge(0))

    def __attrs_post_init__(self):
        assert (self.belief_size > 0) == (self.state_size > 0)


def transition_model_parser(embedding_size, hidden_size,
                            x: TransitionPartSizeSpec, y: TransitionPartSizeSpec, z: TransitionPartSizeSpec,
                            dense_activation_fn, min_stddev, *, env: 'AutoResetEnvBase'):
    return TransitionModel(
        x.belief_size,
        x.state_size,
        y.belief_size,
        y.state_size,
        z.belief_size,
        z.state_size,
        env.action_ndim,
        hidden_size,
        embedding_size,
        activation_function=dense_activation_fn,
        min_stddev=min_stddev,
    )


@attrs.define(kw_only=True, auto_attribs=True)
class TransitionModelConfig:
    _target_: str = attrs.Factory(lambda: f"{transition_model_parser.__module__}.{transition_model_parser.__qualname__}")
    _partial_: bool = True

    class InstantiatedT(Protocol):  # for typing
        def __call__(self, *, env: 'AutoResetEnvBase') -> TransitionModel: ...

    dense_activation_fn: ActivationKind = ActivationKind.elu
    embedding_size: int = attrs.field(default=1024, validator=attrs.validators.gt(0))  # enc(o_t) intermediate (before MLP) size
    hidden_size: int = attrs.field(default=200, validator=attrs.validators.gt(0))
    x: TransitionPartSizeSpec = attrs.Factory(lambda: TransitionPartSizeSpec(belief_size=120, state_size=20))
    y: TransitionPartSizeSpec = attrs.Factory(lambda: TransitionPartSizeSpec(belief_size=120, state_size=20))
    z: TransitionPartSizeSpec = attrs.Factory(lambda: TransitionPartSizeSpec(belief_size=0, state_size=0))
    min_stddev: float = attrs.field(default=0.1, validator=attrs.validators.gt(0))



def reward_model_parser(dense_activation_fn, hidden_size, stddev, *,
                        transition_model: TransitionModel):
    return RewardModel(
        transition_model.x_belief_size,
        transition_model.x_state_size,
        transition_model.y_belief_size,
        transition_model.y_state_size,
        hidden_size,
        stddev=stddev,
        activation_function=dense_activation_fn,
    )


@attrs.define(kw_only=True, auto_attribs=True)
class RewardModelConfig:
    _target_: str = attrs.Factory(lambda: f"{reward_model_parser.__module__}.{reward_model_parser.__qualname__}")
    _partial_: bool = True

    class InstantiatedT(Protocol):  # for typing
        def __call__(self, *, transition_model: TransitionModel) -> RewardModel: ...

    dense_activation_fn: ActivationKind = ActivationKind.elu
    hidden_size: int = attrs.field(default=400, validator=attrs.validators.gt(0))
    stddev: float = attrs.field(default=1, validator=attrs.validators.gt(0))


def observation_model_parser(cnn_activation_fn, hidden_size, stddev, filter_base: int, *,
                             env: 'AutoResetEnvBase', transition_model: TransitionModel):
    return ObservationModel(
        env.observation_shape,
        transition_model.x_belief_size + transition_model.y_belief_size + transition_model.z_belief_size,
        transition_model.x_state_size + transition_model.y_state_size + transition_model.z_state_size,
        hidden_size=hidden_size,
        activation_function=cnn_activation_fn,
        stddev=stddev,
        filter_base=filter_base,
    )


@attrs.define(kw_only=True, auto_attribs=True)
class ObservationModelConfig:
    _target_: str = attrs.Factory(lambda: f"{observation_model_parser.__module__}.{observation_model_parser.__qualname__}")
    _partial_: bool = True

    class InstantiatedT(Protocol):  # for typing
        def __call__(self, *, env: 'AutoResetEnvBase', transition_model: TransitionModel) -> ObservationModel: ...

    cnn_activation_fn: ActivationKind = ActivationKind.relu
    hidden_size: int = attrs.field(default=1024, validator=attrs.validators.gt(0))
    filter_base: int = attrs.field(default=32, validator=attrs.validators.gt(0))
    stddev: float = attrs.field(default=1, validator=attrs.validators.gt(0))


def actor_model_parser(dense_activation_fn, hidden_size, *,
                       env: 'AutoResetEnvBase', transition_model: TransitionModel):
    return ActorModel(
        transition_model.x_belief_size,
        transition_model.x_state_size,
        hidden_size,
        env.action_ndim,
        dense_activation_fn,
    )


@attrs.define(kw_only=True, auto_attribs=True)
class ActorModelConfig:
    _target_: str = attrs.Factory(lambda: f"{actor_model_parser.__module__}.{actor_model_parser.__qualname__}")
    _partial_: bool = True

    class InstantiatedT(Protocol):  # for typing
        def __call__(self, *, env: 'AutoResetEnvBase', transition_model: TransitionModel) -> ActorModel: ...

    dense_activation_fn: ActivationKind = ActivationKind.elu
    hidden_size: int = attrs.field(default=400, validator=attrs.validators.gt(0))


def encoder_model_parser(transition_model_or_dim: Union[TransitionModel, int], *,
                         cnn_activation_fn, filter_base: int, env: 'AutoResetEnvBase'):
    if isinstance(transition_model_or_dim, TransitionModel):
        transition_model_or_dim = transition_model_or_dim.embedding_size
    return EncoderModel(
        env.observation_shape,
        transition_model_or_dim,
        activation_function=cnn_activation_fn,
        filter_base=filter_base,
    )


@attrs.define(kw_only=True, auto_attribs=True)
class EncoderModelConfig:
    _target_: str = attrs.Factory(lambda: f"{encoder_model_parser.__module__}.{encoder_model_parser.__qualname__}")
    _partial_: bool = True

    class InstantiatedT(Protocol):  # for typing
        def __call__(self, transition_model_or_dim: Union[TransitionModel, int], *, env: 'AutoResetEnvBase') -> EncoderModel: ...

    cnn_activation_fn: ActivationKind = ActivationKind.relu
    filter_base: int = attrs.field(default=32, validator=attrs.validators.gt(0))


class DenoisedMDP(AgentBase[TransitionLatentState]):

    @attrs.define(kw_only=True, auto_attribs=True)
    class Config:
        _target_: str = attrs.Factory(lambda: f"{DenoisedMDP.__module__}.{DenoisedMDP.__qualname__}")
        _partial_: bool = True

        class InstantiatedT(Protocol):  # for typing
            def __call__(self, *, env: 'AutoResetEnvBase') -> 'DenoisedMDP': ...

        transition: TransitionModelConfig = attrs.Factory(TransitionModelConfig)
        reward: RewardModelConfig = attrs.Factory(RewardModelConfig)
        encoder: EncoderModelConfig = attrs.Factory(EncoderModelConfig)
        observation: ObservationModelConfig = attrs.Factory(ObservationModelConfig)
        actor: ActorModelConfig = attrs.Factory(ActorModelConfig)

    LatentState = TransitionLatentState

    def init_latent_state(self, *, batch_shape: Tuple[int, ...] = ()) -> LatentState:
        return self.transition_model.init_latent_state(batch_shape=batch_shape)

    def action_from_init_latent_state(self, *, batch_shape: Tuple[int, ...] = (), device=None) -> torch.Tensor:
        if device is None:
            device = self.device
        return torch.zeros((), device=device).expand(*batch_shape, self.actor_model.action_size)

    transition_model: TransitionModel
    reward_model: RewardModel
    encoder_model: EncoderModel
    observation_model: ObservationModel
    actor_model: ActorModel

    def model_learning_parameters(self) -> Iterable[torch.nn.Parameter]:
        yield from self.transition_model.parameters()
        yield from self.reward_model.parameters()
        yield from self.encoder_model.parameters()
        yield from self.observation_model.parameters()

    def __init__(self, transition: TransitionModelConfig.InstantiatedT,
                 reward: RewardModelConfig.InstantiatedT,
                 encoder: EncoderModelConfig.InstantiatedT,
                 observation: ObservationModelConfig.InstantiatedT,
                 actor: ActorModelConfig.InstantiatedT, *, env: 'AutoResetEnvBase'):  # value,
        super().__init__()
        self.transition_model = transition(env=env)
        self.reward_model = reward(transition_model=self.transition_model)
        self.encoder_model = encoder(env=env, transition_model_or_dim=self.transition_model)
        self.observation_model = observation(env=env, transition_model=self.transition_model)
        self.actor_model = actor(env=env, transition_model=self.transition_model)

    def train_reconstruct(self, data: 'ExperienceReplay.Data') -> AgentBase.TrainOutput:
        transition_output: TransitionOutputWithPosterior = self.transition_model.posterior_rsample(
            actions=data.action,
            next_observations=self.encoder_model(data.next_observation),
            rewards=data.reward,
            next_observation_nonfirststeps=data.next_observation_nonfirststep,
        )

        reward_prediction, reward_x_prediction_mean = self.reward_model.get_distn_and_x_mean(
            transition_output.posterior_latent_state)
        observation_prediction: torch.distributions.Normal = self.observation_model(
            transition_output.posterior_latent_state)

        return AgentBase.TrainOutput(
            transition_output=transition_output,
            posterior_latent_state=transition_output.posterior_latent_state,
            observation_prediction=observation_prediction,
            reward_prediction=reward_prediction,
            reward_denoised_prediction_mean=reward_x_prediction_mean,
        )

    def imagine_ahead_noiseless(
        self,
        previous_latent_state: LatentState,
        planning_horizon: int = 12, freeze_latent_model: bool = True,
    ) -> AgentBase.ImagineOutput:
        """
        imagine_ahead is the function to draw the imaginary trajectory using the dynamics model, actor, critic.
        Input: current state (posterior), current belief (hidden), policy, transition_model  # torch.Size([50, 30]) torch.Size([50, 200])
        Output: generated trajectory of features includes beliefs, prior_states, prior_means, prior_stddevs
                torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        """

        with contextlib.ExitStack() as stack:
            if freeze_latent_model:
                stack.enter_context(FreezeParameters([self.transition_model, self.reward_model]))

            # ignore batch shape structure
            previous_latent_state = previous_latent_state.new_emptydim(y=True, z=True).flatten().detach()
            latent_states: List[TransitionLatentState] = []

            for _ in range(planning_horizon):
                action = self.actor_model.get_action(previous_latent_state.detach())
                transition_output = self.transition_model.x_prior_rsample_one_step(
                    action,
                    previous_latent_state=previous_latent_state,
                )
                latent_states.append(transition_output.prior_latent_state)
                previous_latent_state = latent_states[-1]

            all_latent_states = TransitionLatentState.stack(latent_states, dim=0)
            imagined_reward: torch.distributions.Normal = self.reward_model(all_latent_states, x_only=True)
            imagined_reward_mean: torch.Tensor = imagined_reward.mean
            return AgentBase.ImagineOutput(all_latent_states, imagined_reward_mean)

    def posterior_rsample_one_step(self, latent_state: LatentState,  # h/s_{t-1}
                                   action: torch.Tensor,  # a_{t-1}
                                   next_observation: torch.Tensor,  # o_{t}
                                   reward: torch.Tensor,  # r_{t}
                                   next_observation_nonfirststep: Optional[torch.Tensor] = None,
                                   ) -> LatentState:
        transition_output = self.transition_model.posterior_rsample_one_step(
            action=action,
            next_observation=self.encoder_model(next_observation),
            reward=reward,
            next_observation_nonfirststep=next_observation_nonfirststep,
            previous_latent_state=latent_state,
        )
        return transition_output.posterior_latent_state

    def convert_latent_state_to_actor_input(self, latent_state: TransitionLatentState) -> TransitionLatentState:
        return latent_state
