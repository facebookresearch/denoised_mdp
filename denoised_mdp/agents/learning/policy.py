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
import attrs
import itertools
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..networks import ActivationKind, ValueModel, ActorModel
from ..base import AgentBase
from ..utils import lambda_return
from .base import OptimizerCtorCallable, BaseLearning

from ...memory import ExperienceReplay


def value_model_parser(dense_activation_fn, hidden_size, *,
                       world_model: AgentBase):
    return ValueModel(
        world_model.transition_model.x_belief_size,
        world_model.transition_model.x_state_size,
        hidden_size,
        dense_activation_fn,
        action_size=None,
    )


@attrs.define(kw_only=True, auto_attribs=True)
class ValueModelConfig:
    _target_: str = attrs.Factory(lambda: f"{value_model_parser.__module__}.{value_model_parser.__qualname__}")
    _partial_: bool = True

    class InstantiatedT(Protocol):  # for typing
        def __call__(self, *, world_model: AgentBase) -> ValueModel: ...

    dense_activation_fn: ActivationKind = ActivationKind.elu
    hidden_size: int = attrs.field(default=400, validator=attrs.validators.gt(0))  # hidden_layer size


def q_model_parser(dense_activation_fn, hidden_size, *,
                   world_model: AgentBase):
    return ValueModel(
        world_model.transition_model.x_belief_size,
        world_model.transition_model.x_state_size,
        hidden_size,
        dense_activation_fn,
        action_size=world_model.actor_model.action_size,
    )


@attrs.define(kw_only=True, auto_attribs=True)
class QModelConfig:
    _target_: str = attrs.Factory(lambda: f"{q_model_parser.__module__}.{q_model_parser.__qualname__}")
    _partial_: bool = True

    class InstantiatedT(Protocol):  # for typing
        def __call__(self, *, world_model: AgentBase) -> ValueModel: ...

    dense_activation_fn: ActivationKind = ActivationKind.elu
    hidden_size: int = attrs.field(default=400, validator=attrs.validators.gt(0))  # hidden_layer size


class BasePolicyLearning(BaseLearning):

    @attrs.define(kw_only=True, auto_attribs=True)
    class Config:
        _kind_: ClassVar[str]
        _target_: str
        _partial_: bool = True

        class InstantiatedT(Protocol):  # for typing
            def __call__(self, *, world_model: AgentBase, optimizer_ctor: OptimizerCtorCallable,
                         actor_grad_clip_norm: Optional[float] = None) -> 'BasePolicyLearning': ...

        discount: float = attrs.field(default=0.99, validator=[attrs.validators.gt(0), attrs.validators.lt(1)])
        actor_lr: float = attrs.field(default=8e-5, validator=attrs.validators.gt(0))
        actor_grad_clip_norm: Optional[float] = attrs.field(default=100, validator=attrs.validators.optional(attrs.validators.gt(0)))

    discount: float

    def __init__(self, discount: float, actor_lr: float, *,
                 world_model: AgentBase, optimizer_ctor: OptimizerCtorCallable,
                 actor_grad_clip_norm: Optional[float] = None):
        super().__init__(optimizer_ctor=optimizer_ctor, device=world_model.device)
        self.discount = discount
        self.add_optimizer(
            'actor', parameters=world_model.actor_model.parameters(),
            lr=actor_lr, grad_clip_norm=actor_grad_clip_norm)

    @abc.abstractmethod
    def train_step(self, data: ExperienceReplay.Data, train_out: Optional[AgentBase.TrainOutput],
                   world_model: AgentBase) -> Dict[str, torch.Tensor]:
        pass


class DynamicsBackpropagateActorCritic(BasePolicyLearning):
    r"""
    Dreamer-style
    """

    @attrs.define(kw_only=True, auto_attribs=True)
    class Config(BasePolicyLearning.Config):
        _kind_: ClassVar[str] = 'dynamics_backprop'
        _target_: str = attrs.Factory(lambda: f"{DynamicsBackpropagateActorCritic.__module__}.{DynamicsBackpropagateActorCritic.__qualname__}")
        planning_horizon: int = attrs.field(default=15, validator=attrs.validators.gt(0))
        lambda_return_discount: float = attrs.field(default=0.95, validator=[attrs.validators.gt(0), attrs.validators.lt(1)])
        value: ValueModelConfig = attrs.Factory(ValueModelConfig)
        value_lr: float = attrs.field(default=8e-5, validator=attrs.validators.gt(0))
        value_grad_clip_norm: Optional[float] = attrs.field(default=100, validator=attrs.validators.optional(attrs.validators.gt(0)))

    plan_horizon_discount: torch.Tensor
    value_model: ValueModel

    def __init__(self, discount: float, actor_lr: float, actor_grad_clip_norm: float,
                 planning_horizon: int, lambda_return_discount: float,
                 value: ValueModelConfig.InstantiatedT,
                 value_lr: float, value_grad_clip_norm: float,
                 *,
                 world_model: AgentBase, optimizer_ctor: OptimizerCtorCallable):
        super().__init__(discount=discount, actor_lr=actor_lr, actor_grad_clip_norm=actor_grad_clip_norm,
                         world_model=world_model, optimizer_ctor=optimizer_ctor)
        self.planning_horizon = planning_horizon
        self.lambda_return_discount = lambda_return_discount

        plan_horizon_discount = torch.ones(self.planning_horizon - 1, 1, device=self.device)
        plan_horizon_discount[1:].fill_(self.discount)
        torch.cumprod(plan_horizon_discount, dim=0, out=plan_horizon_discount)
        self.register_buffer('plan_horizon_discount', plan_horizon_discount)

        self.value_model = value(world_model=world_model).to(self.device)
        self.add_optimizer(
            'value', parameters=self.value_model.parameters(),
            lr=value_lr, grad_clip_norm=value_grad_clip_norm)

    def train_step(self, data: ExperienceReplay.Data, train_out: Optional[AgentBase.TrainOutput],
                   world_model: AgentBase) -> Dict[str, torch.Tensor]:
        assert train_out is not None
        imagine_out = world_model.imagine_ahead_noiseless(
            previous_latent_state=train_out.posterior_latent_state,
            freeze_latent_model=True,
            planning_horizon=self.planning_horizon,
        )
        value_prediction: torch.Tensor = self.value_model(imagine_out.latent_states)
        lambda_return_prediction = lambda_return(
            imagine_out.reward_mean[:-1], value_prediction[:-1], bootstrap=value_prediction[-1],
            discount=self.discount, lambda_=self.lambda_return_discount,
        )

        actor_loss = -(lambda_return_prediction * self.plan_horizon_discount).mean()
        with self.optimizers['actor'].update_context():
            torch.autograd.backward(
                actor_loss,
                retain_graph=True,
                inputs=list(world_model.actor_model.parameters()),
            )

        value_dist = torch.distributions.Normal(value_prediction[:-1], 1)
        value_loss: torch.Tensor = -value_dist.log_prob(lambda_return_prediction.detach())
        value_loss = value_loss.mul(self.plan_horizon_discount).mean()
        with self.optimizers['value'].update_context():
            torch.autograd.backward(
                value_loss,
                inputs=list(self.value_model.parameters()),
            )

        return dict(actor_loss=actor_loss, value_loss=value_loss)


def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.
    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in itertools.zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def polyak_update(
    params: Iterable[torch.nn.Parameter],
    target_params: Iterable[torch.nn.Parameter],
    tau: float,
) -> None:
    r"""
    https://github.com/DLR-RM/stable-baselines3/blob/e24147390d2ce3b39cafc954e079d693a1971330/stable_baselines3/common/utils.py#L410

    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    See https://github.com/DLR-RM/stable-baselines3/issues/93
    :param params: parameters to use to update the target params
    :param target_params: parameters to update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with torch.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip_strict(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)


class RLAlgorithmInput(NamedTuple):
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_observation: torch.Tensor
    nonterminal: Optional[torch.Tensor]


class GeneralRLLearning(BasePolicyLearning):

    @attrs.define(kw_only=True, auto_attribs=True)
    class Config(BasePolicyLearning.Config):
        actor_lr: float = attrs.field(default=3e-4, validator=attrs.validators.gt(0))
        actor_grad_clip_norm: Optional[float] = attrs.field(default=None, validator=attrs.validators.optional(attrs.validators.gt(0)))


    def create_input(self, data: ExperienceReplay.Data, train_out: Optional[AgentBase.TrainOutput]):
        r"""
        Convert `s`-space trajectorcy to `x`-space trajectory
        """
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            assert train_out is not None
            posterior_latent_state = train_out.transition_output.posterior_latent_state
            posterior_latent_state = posterior_latent_state.new_emptydim(y=True, z=True).detach()
            full_obs = posterior_latent_state.x_feature.detach()
            full_rew = train_out.reward_denoised_prediction_mean

            obs = full_obs[:-1].flatten(0, 1)
            next_obs = full_obs[1:].flatten(0, 1)
            act = data.action[1:].flatten(0, 1)
            rew = full_rew[1:].flatten(0, 1)
            # For `nonterminal`, we assume that the env only ends by timing out. So any transition is
            # non-terminal really, except for the potential (a_T, s'_next_traj_0) in DenoisedMDP data.
            if data.next_observation_nonfirststep is None:
                nonterminal = None
            else:
                raise NotImplementedError
            return RLAlgorithmInput(
                observation=obs,
                action=act,
                reward=rew,
                next_observation=next_obs,
                nonterminal=nonterminal,
            )

    @abc.abstractmethod
    def rl_algorithm_step(self, input: RLAlgorithmInput, actor_model: ActorModel) -> Dict[str, torch.Tensor]:
        pass

    def train_step(self, data: ExperienceReplay.Data, train_out: Optional[AgentBase.TrainOutput],
                   world_model: AgentBase) -> Dict[str, torch.Tensor]:
        return self.rl_algorithm_step(
            self.create_input(data, train_out),
            world_model.actor_model,
        )


class SoftActorCritic(GeneralRLLearning):
    r"""
    https://github.com/DLR-RM/stable-baselines3/blob/e24147390d2ce3b39cafc954e079d693a1971330/stable_baselines3/sac/sac.py
    """

    @attrs.define(kw_only=True, auto_attribs=True)
    class Config(GeneralRLLearning.Config):
        _kind_: ClassVar[str] = 'sac'
        _target_: str = attrs.Factory(lambda: f"{SoftActorCritic.__module__}.{SoftActorCritic.__qualname__}")

        target_update_interval: int = attrs.field(default=1, validator=attrs.validators.gt(0))
        tau: float = attrs.field(default=0.005, validator=[attrs.validators.gt(0), attrs.validators.lt(1)])
        q: QModelConfig = attrs.Factory(QModelConfig)
        q_ent_coef_lr: float = attrs.field(default=3e-4, validator=attrs.validators.gt(0))
        # use ent_coef = auto, target_entropy = auto


    plan_horizon_discount: torch.Tensor

    def __init__(self, discount: float,
                 actor_lr: float, actor_grad_clip_norm: Optional[float],
                 target_update_interval: int, tau: float,
                 q: QModelConfig.InstantiatedT, q_ent_coef_lr: float,
                #  imagine_action_noise: float,
                 *,
                 world_model: AgentBase, optimizer_ctor: OptimizerCtorCallable):
        super().__init__(
            discount=discount,
            actor_lr=actor_lr, actor_grad_clip_norm=actor_grad_clip_norm,
            world_model=world_model, optimizer_ctor=optimizer_ctor)
        self.target_update_interval = target_update_interval
        self.tau = tau

        self.q_model_0: ValueModel = q(world_model=world_model).to(self.device)
        self.q_model_1: ValueModel = q(world_model=world_model).to(self.device)

        self.q_model_0_target: ValueModel = q(world_model=world_model).to(self.device)
        self.q_model_0_target.load_state_dict(self.q_model_0.state_dict())
        self.q_model_1_target: ValueModel = q(world_model=world_model).to(self.device)
        self.q_model_1_target.load_state_dict(self.q_model_1.state_dict())

        self.add_optimizer(
            'q',
            parameters=itertools.chain(self.q_model_0.parameters(), self.q_model_1.parameters()),
            lr=q_ent_coef_lr)

        self.log_ent_coef = nn.Parameter(torch.zeros((), device=self.device, requires_grad=True))
        self.target_entropy = -float(world_model.actor_model.action_size)

        self.add_optimizer(
            'ent_coef',
            parameters=[self.log_ent_coef],
            lr=q_ent_coef_lr)

        self.num_steps = 0

    def trainable_parameters(self):
        yield from self.q_model_0.parameters()
        yield from self.q_model_1.parameters()
        yield self.log_ent_coef

    def rl_algorithm_step(self, input: RLAlgorithmInput, actor_model: ActorModel) -> Dict[str, torch.Tensor]:
        act_distn = actor_model.get_action_distn(input.observation)
        on_pi_act: torch.Tensor = act_distn.rsample()
        on_pi_act_log_prob: torch.Tensor = act_distn.log_prob(on_pi_act)

        losses: Dict[str, torch.Tensor] = {}

        ent_coef = self.log_ent_coef.detach().exp()

        # q loss
        with torch.no_grad():
            # target q
            next_act_distn = actor_model.get_action_distn(input.next_observation)
            next_on_pi_act: torch.Tensor = next_act_distn.rsample()
            next_on_pi_act_log_prob: torch.Tensor = next_act_distn.log_prob(on_pi_act)

            next_q = torch.min(
                self.q_model_0_target(input.next_observation, next_on_pi_act),
                self.q_model_1_target(input.next_observation, next_on_pi_act),
            )
            assert next_on_pi_act_log_prob.ndim == next_q.ndim == 1
            next_q = next_q - ent_coef * next_on_pi_act_log_prob
            if input.nonterminal is None:
                target_q = input.reward + self.discount * next_q
            else:
                target_q = input.reward + input.nonterminal * self.discount * next_q
            assert target_q.ndim == 1
            target_q = target_q.detach()

        data_q0 = self.q_model_0(input.observation, input.action)
        data_q1 = self.q_model_1(input.observation, input.action)
        q0_loss = 0.5 * F.mse_loss(data_q0, target_q)
        q1_loss = 0.5 * F.mse_loss(data_q1, target_q)
        with self.optimizers['q'].update_context():
            q0_loss.backward()
            q1_loss.backward()
        losses['q0_loss'] = q0_loss
        losses['q1_loss'] = q1_loss

        # actor loss
        current_q = torch.min(
            self.q_model_0(input.observation, on_pi_act),
            self.q_model_1(input.observation, on_pi_act),
        )
        actor_loss = (ent_coef * on_pi_act_log_prob - current_q)
        assert actor_loss.ndim == 1
        actor_loss = actor_loss.mean()
        with self.optimizers['actor'].update_context():
            actor_loss.backward()
        losses['actor_loss'] = actor_loss

        # ent_coef loss
        with self.optimizers['ent_coef'].update_context():
            ent_coef_loss = -(self.log_ent_coef * (on_pi_act_log_prob.detach() + self.target_entropy)).mean()
            ent_coef_loss.backward()
        losses['ent_coef_loss'] = ent_coef_loss

        # Update target networks
        self.num_steps += 1
        if self.num_steps % self.target_update_interval == 0:
            polyak_update(self.q_model_0.parameters(), self.q_model_0_target.parameters(), self.tau)
            polyak_update(self.q_model_1.parameters(), self.q_model_1_target.parameters(), self.tau)

        return losses
