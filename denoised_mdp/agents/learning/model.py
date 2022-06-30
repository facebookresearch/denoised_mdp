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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..networks import TransitionOutputWithPosterior, TransitionLatentPart
from ..base import AgentBase
from ..denoised_mdp import DenoisedMDP
from .base import OptimizerCtorCallable, BaseLearning

from ...memory import ExperienceReplay


class BaseLoss(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, data: ExperienceReplay.Data, train_out: AgentBase.TrainOutput,
                ) -> Dict[str, Tuple[float, torch.Tensor]]:
        pass

    def __call__(self, data: ExperienceReplay.Data, train_out: AgentBase.TrainOutput,
                 ) -> Dict[str, Tuple[float, torch.Tensor]]:
        return super().__call__(data, train_out)


class ObservationLoss(BaseLoss):
    @attrs.define(kw_only=True, auto_attribs=True)
    class Config:
        _target_: str = attrs.Factory(lambda: f"{ObservationLoss.__module__}.{ObservationLoss.__qualname__}")
        weight: float = attrs.field(default=1, validator=attrs.validators.ge(0))

    def __init__(self, weight: float, name: str = 'observation_loss'):
        super().__init__()
        self.weight = weight
        self.name = name

    def forward(self, data: ExperienceReplay.Data, train_out: AgentBase.TrainOutput,
                ) -> Dict[str, Tuple[float, torch.Tensor]]:
        log_prob: torch.Tensor = train_out.observation_prediction.log_prob(data.next_observation)
        loss = -log_prob.flatten(-3, -1).sum(-1).mean()
        return {
            self.name: (self.weight, loss),
            self.name + '_mse': (0, F.mse_loss(train_out.observation_prediction.mean.detach(), data.next_observation)),
        }

    def extra_repr(self) -> str:
        return f"weight={self.weight}"


class RewardLoss(BaseLoss):
    @attrs.define(kw_only=True, auto_attribs=True)
    class Config:
        _target_: str = attrs.Factory(lambda: f"{RewardLoss.__module__}.{RewardLoss.__qualname__}")
        weight: float = attrs.field(default=1, validator=attrs.validators.ge(0))

    def __init__(self, weight: float, name: str = 'reward_loss'):
        super().__init__()
        self.weight = weight
        self.name = name

    def forward(self, data: ExperienceReplay.Data, train_out: AgentBase.TrainOutput,
                ) -> Dict[str, Tuple[float, torch.Tensor]]:
        log_prob: torch.Tensor = train_out.reward_prediction.log_prob(data.reward)
        loss = -log_prob.mean()
        return {
            self.name: (self.weight, loss),
            self.name + '_mse': (0, F.mse_loss(train_out.reward_prediction.mean.detach(), data.reward)),
        }

    def extra_repr(self) -> str:
        return f"weight={self.weight}"


class KLLoss(BaseLoss):
    @attrs.define(frozen=True, kw_only=True, auto_attribs=True)
    class PartPosteriorPriorKLTermSpec:
        part: TransitionLatentPart
        weight: float
        free_nats: float

    @attrs.define(kw_only=True, auto_attribs=True)
    class Config:
        _target_: str = attrs.Factory(lambda: f"{KLLoss.__module__}.{KLLoss.__qualname__}")
        alpha: float = attrs.field(default=1, validator=attrs.validators.ge(0))

        # The paper only experiments a single beta hyperparameter, applied on
        # the y and z terms. Here we provide more freedom to weight all three
        # terms independently. For reproducing the paper results, use beta_x=1
        # (default) and beta_y=beta_z=beta.

        beta_x: float = attrs.field(default=1, validator=[attrs.validators.ge(0), attrs.validators.le(1)])
        free_nats_x: float = attrs.field(default=3, validator=attrs.validators.ge(0))

        # Option to only apply free nats to part of the KL_x term. The default
        # value (None) makes all beta_x use free nats. Setting this to beta
        # performs the alternative KL free nats strategy discussed in B.1.2.
        beta_free_nats_x: Optional[float] = attrs.field(
            default=None,
            validator=attrs.validators.optional(attrs.validators.ge(0)))

        beta_y: float = attrs.field(default=0.25, validator=[attrs.validators.ge(0), attrs.validators.le(1)])
        free_nats_y: float = attrs.field(default=0, validator=attrs.validators.ge(0))

        beta_z: float = attrs.field(default=0.25, validator=[attrs.validators.ge(0), attrs.validators.le(1)])
        free_nats_z: float = attrs.field(default=0, validator=attrs.validators.ge(0))

    def __init__(self, alpha: float,
                 beta_x: float, free_nats_x: float, beta_free_nats_x: Optional[float],
                 beta_y: float, free_nats_y: float,
                 beta_z: float, free_nats_z: float, *,
                 name_format: str = 'kl_{spec}_loss'):
        super().__init__()
        if beta_free_nats_x is None:
            beta_free_nats_x = beta_x
        if beta_free_nats_x > beta_x:
            raise ValueError(f'Expected beta_x ({beta_x:g}) >= beta_free_nats_x ({beta_free_nats_x:g})')
        self.terms = dict(
            x=KLLoss.PartPosteriorPriorKLTermSpec(
                part=TransitionLatentPart.x, weight=alpha * beta_free_nats_x, free_nats=free_nats_x),
            y=KLLoss.PartPosteriorPriorKLTermSpec(
                part=TransitionLatentPart.y, weight=alpha * beta_y, free_nats=free_nats_y),
            z=KLLoss.PartPosteriorPriorKLTermSpec(
                part=TransitionLatentPart.z, weight=alpha * beta_z, free_nats=free_nats_z),
        )
        if beta_free_nats_x < beta_x:
            self.terms.update(
                x_no_free_nats=KLLoss.PartPosteriorPriorKLTermSpec(
                    part=TransitionLatentPart.x, weight=alpha * (beta_x - beta_free_nats_x), free_nats=0),
            )
        self.name_format = name_format

    def _compute_losses(self, transition_output: TransitionOutputWithPosterior):
        losses = {}
        for spec, term in self.terms.items():
            loss: torch.Tensor = torch.distributions.kl_divergence(
                transition_output.posterior[term.part],
                transition_output.prior[term.part],
            ).sum(dim=-1)  # isotropic Gaussians. sum to get multivariate kl
            if term.free_nats > 0:
                loss = loss.clamp(min=term.free_nats)
            losses[self.name_format.format(spec=spec)] = (term.weight, loss.mean())
        return losses

    def forward(self, data: ExperienceReplay.Data, train_out: AgentBase.TrainOutput,
                ) -> Dict[str, Tuple[float, torch.Tensor]]:
        return self._compute_losses(train_out.transition_output)

    def extra_repr(self) -> str:
        reprs = []
        for spec, term in self.terms.items():
            reprs.append(f'{spec}={term}')
        return '\n'.join(reprs)


# workaround https://github.com/omry/omegaconf/issues/963
KLLoss.Config = attrs.resolve_types(KLLoss.Config)


class BaseModelLearning(BaseLearning):
    def __init__(self, *, world_model: AgentBase, optimizer_ctor: OptimizerCtorCallable):
        super().__init__(optimizer_ctor=optimizer_ctor, device=world_model.device)

    @abc.abstractmethod
    def train_step(self, data: ExperienceReplay.Data, world_model: AgentBase,
                   grad_update: bool = True) -> Tuple[Optional[AgentBase.TrainOutput], Dict[str, torch.Tensor]]:
        pass


class VariationalModelLearning(BaseModelLearning):

    @attrs.define(kw_only=True, auto_attribs=True)
    class Config:
        _target_: str = attrs.Factory(lambda: f"{VariationalModelLearning.__module__}.{VariationalModelLearning.__qualname__}")
        _partial_: bool = True

        class InstantiatedT(Protocol):  # for typing
            def __call__(self, *, world_model: AgentBase, optimizer_ctor: OptimizerCtorCallable) -> 'VariationalModelLearning': ...

        observation: ObservationLoss.Config = attrs.Factory(ObservationLoss.Config)
        reward: RewardLoss.Config = attrs.Factory(RewardLoss.Config)
        kl: KLLoss.Config = attrs.Factory(KLLoss.Config)
        lr: float = attrs.field(default=6e-4, validator=attrs.validators.gt(0))
        grad_clip_norm: Optional[float] = attrs.field(default=100, validator=attrs.validators.optional(attrs.validators.gt(0)))


    losses: nn.ModuleDict
    lr: float

    def __init__(self, observation, reward, kl, lr: float, grad_clip_norm: float, *,
                 world_model: AgentBase, optimizer_ctor: OptimizerCtorCallable):
        super().__init__(world_model=world_model, optimizer_ctor=optimizer_ctor)
        assert isinstance(world_model, DenoisedMDP)
        self.losses = nn.ModuleDict(dict(
            observation=observation,
            reward=reward,
            kl=kl,
        ))
        self.add_optimizer(
            'model', parameters=world_model.model_learning_parameters(),
            lr=lr, grad_clip_norm=grad_clip_norm)
        self.lr = lr

    def compute_losses(self, data: ExperienceReplay.Data, train_out: AgentBase.TrainOutput,
                      ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total_loss: Union[float, torch.Tensor] = 0
        losses = {}
        lossm: BaseLoss
        for lossm in self.losses.values():
            terms = lossm(data, train_out)
            for k, (w, l) in terms.items():
                assert isinstance(l, (int, float)) or l.ndim == 0
                if w != 0:
                    total_loss += w * l
                assert k not in losses
                losses[k] = l
        return cast(torch.Tensor, total_loss), losses

    def train_step(self, data: ExperienceReplay.Data, world_model: AgentBase,
                   grad_update: bool = True) -> Tuple[Optional[AgentBase.TrainOutput], Dict[str, torch.Tensor]]:
        if grad_update:
            train_out = world_model.train_reconstruct(data)
            total_loss, losses = self.compute_losses(data, train_out)
            assert 'total_loss' not in losses
            losses.update(total_loss=total_loss)
            with self.optimizers['model'].update_context():
                torch.autograd.backward(total_loss, inputs=list(world_model.model_learning_parameters()))
        else:
            train_out = None
            losses = {}
        return train_out, losses
