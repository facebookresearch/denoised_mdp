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

import torch
import torch.nn as nn

from ..utils import OptimWrapper


class OptimizerCtorCallable(Protocol):
    def __call__(self, parameters: Iterable[nn.Parameter], lr: float) -> torch.optim.Optimizer: ...


# Represents an "algorithm" for a certain objective
class BaseLearning(nn.Module, metaclass=abc.ABCMeta):
    options = []

    optimizers: Dict[str, OptimWrapper]  # OptimWrapper makes stepping optimizers easier-to-write
    device: torch.device

    def __init__(self, *, optimizer_ctor: OptimizerCtorCallable, device: torch.device):
        super().__init__()
        self.optimizer_ctor = optimizer_ctor
        self.optimizers = dict()
        self.device = device

    def add_optimizer(self, name: str, parameters: Iterable[nn.Parameter],
                      lr: float, *, grad_clip_norm: Optional[float] = None):
        assert name not in self.optimizers
        parameters = tuple(parameters)
        assert all(p.device == self.device for p in parameters)
        self.optimizers[name] = OptimWrapper(
            self.optimizer_ctor(parameters, lr=lr),
            grad_clip_norm=grad_clip_norm,
        )

    def trainable_parameters(self):
        yield from self.parameters()

    def named_optimizers(self) -> Iterable[Tuple[str, OptimWrapper]]:
        yield from self.optimizers.items()

    def state_dict(self):
        return dict(
            module=super().state_dict(),
            optimizers={
                k: optim.state_dict() for k, optim in self.named_optimizers()
            },
        )

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict['module'])
        for k, optim in self.named_optimizers():
            optim.load_state_dict(state_dict['optimizers'][k])

    @abc.abstractmethod
    def train_step(self, *args, **kwargs):
        pass
