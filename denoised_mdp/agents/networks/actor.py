# Copyright (c) 2019 Kai Arulkumaran (Original PlaNet parts) Copyright (c) 2020 Yusuke Urakami (Dreamer parts)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import numpy as np

import torch
from torch import jit, nn
from torch.nn import functional as F

import torch.distributions
from torch.distributions.normal import Normal
from torch.distributions.transforms import TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution

from .transition import LatentState as TransitionLatentState
from .utils import BottledModule, get_activation_module, ClipGradTanhBijector


class SampleDist:
    def __init__(self, dist: torch.distributions.Distribution, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        dist: torch.distributions.Distribution = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        dist: torch.distributions.Distribution = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample: torch.Tensor = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist: torch.distributions.Distribution = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample: torch.Tensor = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()

    def rsample(self):
        return self._dist.rsample()

    def log_prob(self, value):
        return self._dist.log_prob(value)


class ActorModel(jit.ScriptModule):
    __constants__ = ['action_size']

    action_size: int

    def __init__(
        self,
        belief_size,
        state_size,
        hidden_size,
        action_size,
        activation_function="elu",
        min_std=1e-4,
        init_std=5,
        mean_scale=5,
    ):
        super().__init__()
        self.net = BottledModule(nn.Sequential(
            nn.Linear(belief_size + state_size, hidden_size),
            get_activation_module(activation_function),
            nn.Linear(hidden_size, hidden_size),
            get_activation_module(activation_function),
            nn.Linear(hidden_size, hidden_size),
            get_activation_module(activation_function),
            nn.Linear(hidden_size, hidden_size),
            get_activation_module(activation_function),
            nn.Linear(hidden_size, 2 * action_size),
        ))
        self.modules = [self.net]

        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale
        self._raw_init_std = np.log(np.exp(self._init_std) - 1)
        self.action_size = action_size

    @jit.script_method
    def forward(self, input_tensor: torch.Tensor):
        action = self.net(input_tensor)

        action_mean, action_stddev = torch.chunk(action, 2, dim=-1)
        action_mean = self._mean_scale * torch.tanh(action_mean / self._mean_scale)
        action_std = F.softplus(action_stddev + self._raw_init_std) + self._min_std
        return action_mean, action_std

    def get_action_distn(self, input: Union[torch.Tensor, TransitionLatentState]) -> torch.distributions.Distribution:
        if isinstance(input, TransitionLatentState):
            input = input.x_feature
        action_mean, action_std = self(input)
        dist: torch.distributions.Distribution = Normal(action_mean, action_std)
        dist = TransformedDistribution(dist, ClipGradTanhBijector())
        return torch.distributions.Independent(dist, 1)

    def get_action(self, input: Union[torch.Tensor, TransitionLatentState], det=False):
        sample_dist = SampleDist(self.get_action_distn(input))
        if det:
            return sample_dist.mode()
        else:
            return sample_dist.rsample()
