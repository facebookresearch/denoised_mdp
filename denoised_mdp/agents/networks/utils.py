# Copyright (c) 2019 Kai Arulkumaran (Original PlaNet parts) Copyright (c) 2020 Yusuke Urakami (Dreamer parts)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import contextlib
import enum

import numpy as np
import torch
from torch import jit, nn
import torch.nn.functional as F
import torch.distributions
from torch.distributions.transforms import TanhTransform



class ActivationKind(enum.Enum):
    relu = nn.ReLU()
    elu = nn.ELU()
    tanh = nn.Tanh()
    sigmoid = nn.Sigmoid()


def get_activation_module(activation_function) -> nn.Module:
    if isinstance(activation_function, str):
        return ActivationKind[activation_function].value
    elif isinstance(activation_function, nn.Module):
        return activation_function
    else:
        assert isinstance(activation_function, ActivationKind)
        return activation_function.value


# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f: Callable[[Any], torch.Tensor], x_tuple: Tuple[torch.Tensor]):
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*[x.flatten(0, 1) for x in x_tuple])
    y_size = y.size()
    output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
    return output


class BottledModule(jit.ScriptModule):
    __constants__ = ["input_elem_dim"]

    def __init__(self, module, input_elem_dim=1):
        super().__init__()
        self.module = module
        self.input_elem_dim = input_elem_dim

    @jit.script_method
    def forward(self, x: torch.Tensor):
        if x.ndim == self.input_elem_dim:
            return self.module(x[None]).squeeze(0)
        else:
            bshape = x.shape[:-self.input_elem_dim]
            return self.module(x.flatten(0, -self.input_elem_dim - 1)).unflatten(0, bshape)


# "atanh", "TanhBijector" and "SampleDist" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
class ClipGradTanhBijector(torch.distributions.Transform):
    domain = TanhTransform.domain
    codomain = TanhTransform.codomain

    def __init__(self):
        super().__init__()
        self.bijective = True

    @property
    def sign(self):
        return 1.0

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where(torch.abs(y) <= 1.0, torch.clamp(y, -0.99999997, 0.99999997), y)
        y = torch.atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (np.log(2) - x - F.softplus(-2.0 * x))


@contextlib.contextmanager
def no_jit_fuser():
    old_cpu_fuse = torch._C._jit_can_fuse_on_cpu()
    old_gpu_fuse = torch._C._jit_can_fuse_on_gpu()
    old_texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
    old_nvfuser_state = torch._C._jit_nvfuser_enabled()
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
    try:
        yield
    finally:
        # recover the previous values
        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fuse)
        torch._C._jit_override_can_fuse_on_gpu(old_gpu_fuse)
        torch._C._jit_set_texpr_fuser_enabled(old_texpr_fuser_state)
        torch._C._jit_set_nvfuser_enabled(old_nvfuser_state)
