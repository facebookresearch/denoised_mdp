# Copyright (c) 2019 Kai Arulkumaran (Original PlaNet parts) Copyright (c) 2020 Yusuke Urakami (Dreamer parts)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import torch
from torch import jit, nn
import torch.nn.functional as F

from .transition import LatentState as TransitionLatentState
from .utils import BottledModule, get_activation_module


class RewardMeanNetwork(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, activation_function="relu"):
        super().__init__()
        self.net = BottledModule(nn.Sequential(
            nn.Linear(belief_size + state_size, hidden_size),
            get_activation_module(activation_function),
            nn.Linear(hidden_size, hidden_size),
            get_activation_module(activation_function),
            nn.Linear(hidden_size, 1),
        ))

    @jit.script_method
    def forward_feature(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.forward_feature(feature)
        return out.squeeze(-1)


class RewardModel(nn.Module):
    stddev: float

    def __init__(self,
                 x_belief_size, x_state_size,
                 y_belief_size, y_state_size, hidden_size,
                 activation_function="relu", stddev=1):
        super().__init__()
        assert (y_belief_size > 0) == (y_state_size > 0)
        self.x_network = RewardMeanNetwork(
            x_belief_size, x_state_size, hidden_size, activation_function)
        if y_state_size > 0:
            self.y_network = RewardMeanNetwork(
                y_belief_size, y_state_size, hidden_size, activation_function)
        else:
            self.add_module('y_network', None)
        self.stddev = stddev

    def get_distn_and_x_mean(self, latent_state: TransitionLatentState,
                             x_only: bool = False) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        mean = x_mean = self.x_network(latent_state.x_feature)
        if not x_only and self.y_network is not None:
            y_mean = self.y_network(latent_state.y_feature)
            mean = mean + y_mean
        stddev = self.stddev / 2 if x_only else self.stddev
        return torch.distributions.Normal(mean, stddev), x_mean

    def forward(self, latent_state: TransitionLatentState, x_only: bool = False) -> torch.distributions.Normal:
        return self.get_distn_and_x_mean(latent_state, x_only)[0]

    def __call__(self, latent_state: TransitionLatentState, x_only: bool = False) -> torch.distributions.Normal:
        return super().__call__(latent_state, x_only)
