# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import sys
if sys.version_info < (3, 8):
    from typing_extensions import Protocol

if (3, 9) <= sys.version_info < (3, 11):
    from typing_extensions import NamedTuple

import abc
import contextlib
import dataclasses

import numpy as np
import torch

from tqdm.auto import tqdm

from .abc import AutoResetEnvBase


#################
# Base
#################

class BasicEnvState(NamedTuple):
    r"""
    All information an actor needs to perform an action.
    """

    num_episodes: torch.Tensor
    num_steps: torch.Tensor
    next_observation: torch.Tensor  # o_{t+1} if not done, reset() if done
    reward: torch.Tensor
    next_observation_is_first_step: Union[bool, torch.Tensor]  # whether the action to make at o_{t+1} is the first step
    info: Optional[AutoResetEnvBase.Info] = None  # maybe store something


class StateProtocol(Protocol):
    basic_env_state: BasicEnvState


StateT = TypeVar('StateT', bound=StateProtocol)


class EnvInteractData(NamedTuple, Generic[StateT]):  # thank god we py37 https://stackoverflow.com/a/50531189
    r"""
    All information about the last step taken.
    """

    @property
    def num_episodes(self) -> torch.Tensor:
        return self.state_after_step.basic_env_state.num_episodes

    @property
    def num_steps(self) -> torch.Tensor:
        return self.state_after_step.basic_env_state.num_steps

    num_new_steps: torch.Tensor  # usually just numel of batch shape

    batch_shape: torch.Size

    is_first_step: Union[bool, torch.Tensor]

    observation: torch.Tensor  # o_t

    action: torch.Tensor  # a_t

    @property
    def reward(self) -> torch.Tensor:
        # r_t
        return self.state_after_step.basic_env_state.reward

    @property
    def done(self) -> Union[bool, torch.Tensor]:
        # in other words, whether the action to make at o_{t+1} is the first step
        return self.state_after_step.basic_env_state.next_observation_is_first_step

    observation_before_reset: torch.Tensor # o_{t+1} if done, else can be garbage

    @property
    def next_observation(self) -> torch.Tensor:
        # o_{t+1} if not done, reset() if done
        return self.state_after_step.basic_env_state.next_observation

    state_after_step: StateT

    last_info: Optional[AutoResetEnvBase.Info] = None

    @property
    def info(self) -> Optional[AutoResetEnvBase.Info]:
        # maybe store something
        return self.state_after_step.basic_env_state.info


class Interactor(abc.ABC, Generic[StateT]):
    def __init__(self, env: AutoResetEnvBase) -> None:
        super().__init__()
        self.env = env

    @abc.abstractmethod
    def init_state(self, first_basic_env_state: BasicEnvState) -> StateT:
        pass

    @abc.abstractmethod
    def act_and_update_state(self, state: StateT) -> Tuple[torch.Tensor, Callable[[BasicEnvState], StateT]]:
        pass


def env_interact(env: AutoResetEnvBase, max_num_steps: Optional[int],
                 interactor_fn: Callable[[AutoResetEnvBase], Interactor[StateT]], *,
                 state: Optional[StateT] = None, tqdm_desc: Optional[str] = 'Interact', track_env_info: bool = False):

    env_batch_shape = env.batch_shape
    interactor = interactor_fn(env)

    if state is None:
        observation, info = env.reset()
        state = interactor.init_state(first_basic_env_state=BasicEnvState(
            num_episodes=torch.zeros(env_batch_shape, dtype=torch.int64),
            num_steps=torch.zeros(env_batch_shape, dtype=torch.int64),
            next_observation=observation,
            reward=torch.zeros(env_batch_shape),
            next_observation_is_first_step=True,
            info=info if track_env_info else None,
        ))

    with tqdm(desc=tqdm_desc, disable=(tqdm_desc is None), total=max_num_steps) as pbar:

        if max_num_steps is None:
            num_steps_upper_bound = np.inf
        else:
            num_steps_upper_bound = max_num_steps

        while (state.basic_env_state.num_steps < num_steps_upper_bound).all():
            basic_env_state = state.basic_env_state
            # update to be before this action, then act
            is_first_step = basic_env_state.next_observation_is_first_step
            observation = basic_env_state.next_observation
            last_info = basic_env_state.info
            action, next_state_fn = interactor.act_and_update_state(state)
            # interact
            next_observation, reward, done, info = env.step(action)
            # update state
            actual_env_steps_taken = torch.as_tensor(info.actual_env_steps_taken, dtype=torch.int64)
            basic_env_state = BasicEnvState(
                num_episodes=basic_env_state.num_episodes + torch.as_tensor(done, dtype=torch.int64),
                num_steps=basic_env_state.num_steps + actual_env_steps_taken,
                next_observation=next_observation,
                reward=reward,
                next_observation_is_first_step=done,
                info=info if track_env_info else None,
            )
            pbar.update(
                torch.as_tensor(basic_env_state.num_steps, dtype=torch.int64).min().item() - pbar.n)
            pbar.refresh()
            next_state = next_state_fn(basic_env_state)
            # yield data
            yield EnvInteractData(
                num_new_steps=actual_env_steps_taken,
                batch_shape=env_batch_shape,
                is_first_step=is_first_step,
                observation=observation,
                action=action,
                observation_before_reset=info.observation_before_reset,
                state_after_step=next_state,
                last_info=last_info,
            )
            # prepare for next
            state = next_state


#################
# Random Actor
#################

class RandomActorState(NamedTuple):
    basic_env_state: BasicEnvState


class RandomActorInteractor(Interactor[RandomActorState]):
    def init_state(self, first_basic_env_state: BasicEnvState) -> RandomActorState:
        return RandomActorState(first_basic_env_state)

    def act_and_update_state(self, state: RandomActorState) -> Tuple[torch.Tensor, Callable[[BasicEnvState], RandomActorState]]:
        return self.env.sample_random_action(), RandomActorState


def env_interact_random_actor(env: AutoResetEnvBase, max_num_steps: Optional[int], **kwargs):
    yield from env_interact(env, max_num_steps, interactor_fn=RandomActorInteractor, **kwargs)


################
# Model Actor
################


from ..agents.base import AgentBase, LatentStateT


@dataclasses.dataclass
class ModelActorState(Generic[LatentStateT]):
    basic_env_state: BasicEnvState
    flat_model_latent_state_before_next_observation: LatentStateT  # h/s_t, which leads to a_t
    flat_action_for_model: torch.Tensor  # a_t, which leads to o_{t+1}

    @property
    def flat_next_observation_nonfirststep_tensor(self) -> Optional[torch.Tensor]:  # None = True (i.e., not first step)
        env_batch_numel = self.flat_action_for_model.shape[:-1].numel()
        if env_batch_numel == 1 and self.basic_env_state.next_observation_is_first_step is False:
            return None
        else:
            # NB: ~ is **bitwise** negation: ~True = -2 !
            #     But for tensors, it is logic negation :/, so as_tensor first.
            return torch.as_tensor(
                ~torch.as_tensor(self.basic_env_state.next_observation_is_first_step, dtype=torch.bool),
                dtype=torch.float32,
                device=self.flat_action_for_model.device,
            ).reshape(-1).expand(env_batch_numel)


class ModelActorInteractor(Interactor[ModelActorState[LatentStateT]]):
    def __init__(self, env: AutoResetEnvBase, world_model: AgentBase[LatentStateT], actor_kwargs=dict(),
                 train=False) -> None:
        super().__init__(env)
        self.env_batch_shape = env.batch_shape
        self.world_model = world_model
        self.actor_kwargs = actor_kwargs
        self.train = train

    def init_state(self, first_basic_env_state: BasicEnvState) -> ModelActorState[LatentStateT]:
        return ModelActorState(
            basic_env_state=first_basic_env_state,
            flat_model_latent_state_before_next_observation=self.world_model.init_latent_state(
                batch_shape=(self.env_batch_shape.numel(),),
            ),
            flat_action_for_model=self.world_model.action_from_init_latent_state(
                batch_shape=(self.env_batch_shape.numel(),)),
        )

    @contextlib.contextmanager
    def specified_train_mode(self):
        flag = self.world_model.training
        self.world_model.train(self.train)
        yield
        self.world_model.train(flag)

    def act_and_update_state(self, state: ModelActorState[LatentStateT]) -> Tuple[torch.Tensor, Callable[[BasicEnvState], ModelActorState[LatentStateT]]]:
        basic_env_state: BasicEnvState = state.basic_env_state
        with torch.no_grad(), self.specified_train_mode():
            processed_next_observation = self.env.process_observation_as_network_input(
                basic_env_state.next_observation.to(device=self.world_model.device).view(-1, *self.env.observation_shape),
            )
            # update latent to be before this action
            next_model_latent_state: LatentStateT = self.world_model.posterior_rsample_one_step(
                state.flat_model_latent_state_before_next_observation,
                state.flat_action_for_model,
                next_observation=processed_next_observation,
                reward=torch.as_tensor(basic_env_state.reward, dtype=torch.float32, device=self.world_model.device).view(-1),
                next_observation_nonfirststep=state.flat_next_observation_nonfirststep_tensor,
            ).detach()
            # now we move to next step decision
            # get action
            flat_action_for_model = self.world_model.act(next_model_latent_state, **self.actor_kwargs)
            rebatched_cpu_action = flat_action_for_model.cpu().view(*self.env_batch_shape, -1)

        def get_next_state(next_basic_env_state: BasicEnvState):
            return ModelActorState(
                basic_env_state=next_basic_env_state,
                flat_model_latent_state_before_next_observation=next_model_latent_state,
                flat_action_for_model=flat_action_for_model,
            )

        return rebatched_cpu_action, get_next_state


def env_interact_with_model(env: AutoResetEnvBase, world_model: AgentBase,
                            max_num_steps: Optional[int], actor_kwargs=dict(),
                            train=False, *, state: Optional[ModelActorState] = None,
                            **kwargs):
    yield from env_interact(
        env, max_num_steps,
        interactor_fn=lambda env: ModelActorInteractor(env, world_model, actor_kwargs, train),
        state=state,
        **kwargs,
    )
