# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import sys
if sys.version_info < (3, 8):
    from typing_extensions import Literal

import contextlib

import torch
from torch.nn import Module


@torch.jit.script
def lambda_return(imged_reward: torch.Tensor, value_pred: torch.Tensor, bootstrap: torch.Tensor,
                  discount: float = 0.99, lambda_: float = 0.95) -> torch.Tensor:
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    next_values = torch.cat([value_pred[1:], bootstrap[None]], dim=0)
    inputs = imged_reward + (discount * (1 - lambda_)) * next_values
    lambda_discount: float = discount * lambda_
    last = bootstrap
    outputs: List[torch.Tensor] = []
    for index in range(inputs.shape[0] - 1, -1, -1):
        last = inputs[index] + lambda_discount * last
        outputs.insert(0, last)
    returns = torch.stack(outputs, 0)
    return returns


@contextlib.contextmanager
def training_mode(*ms: Module, training: bool):
    olds = [m.training for m in ms]
    try:
        for m in ms:
            m.train(training)
        yield
    finally:
        for old, m in zip(olds, ms):
            m.train(old)


@contextlib.contextmanager
def optim_step(optimizer: torch.optim.Optimizer, *, grad_clip_norm: Optional[float] = None):
    optimizer.zero_grad()
    yield
    if grad_clip_norm is not None:
        assert len(optimizer.param_groups) == 1
        torch.nn.utils.clip_grad_norm_(
            optimizer.param_groups[0]['params'],
            grad_clip_norm,
            norm_type=2,
        )
    optimizer.step()


class OptimWrapper(object):
    def __init__(self, optim: torch.optim.Optimizer, *, grad_clip_norm: Optional[float] = None):
        self.optim = optim
        self.grad_clip_norm = grad_clip_norm
        if grad_clip_norm is not None:
            assert len(optim.param_groups) == 1

    @contextlib.contextmanager
    def update_context(self):
        self.optim.zero_grad()
        yield
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.param_groups[0]['params'],
                self.grad_clip_norm,
                norm_type=2,
            )
        self.optim.step()

    @property
    def param_groups(self):
        return self.optim.param_groups

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict: dict):
        return self.optim.load_state_dict(state_dict)


class ActivateParameters:
    def __init__(self, modules: Iterable[Optional[Module]]):
        """
      Context manager to locally Activate the gradients.
      example:
      ```
      with ActivateParameters([module]):
          output_tensor = module(input_tensor)
      ```
      :param modules: iterable of modules. used to call .parameters() to freeze gradients.
      """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            # print(param.requires_grad)
            param.requires_grad = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


# "get_parameters" and "FreezeParameters" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def get_parameters(modules: Iterable[Optional[Module]]) -> List[torch.nn.Parameter]:
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        if module is not None:
            model_parameters += list(module.parameters())
    return model_parameters


class FreezeParameters:
    def __init__(self, modules: Iterable[Optional[Module]]):
        """
      Context manager to locally freeze gradients.
      In some cases with can speed up computation because gradients aren't calculated for these listed modules.
      example:
      ```
      with FreezeParameters([module]):
          output_tensor = module(input_tensor)
      ```
      :param modules: iterable of modules. used to call .parameters() to freeze gradients.
      """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]
