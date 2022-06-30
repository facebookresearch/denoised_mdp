# Copyright (c) 2019 Kai Arulkumaran (Original PlaNet parts) Copyright (c) 2020 Yusuke Urakami (Dreamer parts)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import jit, nn

from .transition import LatentState as TransitionLatentState
from .utils import BottledModule, get_activation_module


class ObservationModel(jit.ScriptModule):
    __constants__ = ["hidden_size", "stddev", "min_stddev"]

    stddev: float
    min_stddev: float

    def __init__(self, observation_shape, belief_size, state_size, hidden_size,
                 activation_function="relu", stddev=1, min_stddev=0.1, filter_base=32):
        super().__init__()
        self.hidden_size = hidden_size
        assert len(observation_shape) == 3 and observation_shape[1] == observation_shape[2]
        assert observation_shape[1] in {64, 96}
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        out_nc = observation_shape[0]
        if observation_shape[1] == 64:
            conv_part = [
                get_activation_module(activation_function),
                nn.ConvTranspose2d(hidden_size, filter_base * 4, 5, stride=2),
                get_activation_module(activation_function),
                nn.ConvTranspose2d(filter_base * 4, filter_base * 2, 5, stride=2),
                get_activation_module(activation_function),
                nn.ConvTranspose2d(filter_base * 2, filter_base, 6, stride=2),
                get_activation_module(activation_function),
                nn.ConvTranspose2d(filter_base, out_nc, 6, stride=2),
            ]
        elif observation_shape[1] == 96:
            conv_part = [
                get_activation_module(activation_function),
                nn.ConvTranspose2d(hidden_size, filter_base * 4, 3, stride=1),
                get_activation_module(activation_function),
                nn.ConvTranspose2d(filter_base * 4, filter_base * 4, 5, stride=2),
                get_activation_module(activation_function),
                nn.ConvTranspose2d(filter_base * 4, filter_base * 2, 5, stride=2),
                get_activation_module(activation_function),
                nn.ConvTranspose2d(filter_base * 2, filter_base, 6, stride=2),
                get_activation_module(activation_function),
                nn.ConvTranspose2d(filter_base, out_nc, 6, stride=2),
            ]
        assert not isinstance(conv_part[0], (nn.ConvTranspose2d, nn.Conv2d))
        assert isinstance(conv_part[-1], (nn.ConvTranspose2d, nn.Conv2d))
        self.conv_head = BottledModule(nn.Sequential(*conv_part), input_elem_dim=3)
        self.stddev = stddev
        self.min_stddev = min_stddev

    @jit.script_method
    def forward_feature(self, feature: torch.Tensor) -> torch.Tensor:
        hidden: torch.Tensor = self.fc1(feature)  # No nonlinearity here
        hidden = hidden.unsqueeze(-1).unsqueeze(-1)
        return self.conv_head(hidden)

    def forward(self, latent_state: TransitionLatentState) -> torch.distributions.Normal:
        out = self.forward_feature(latent_state.full_feature)
        return torch.distributions.Normal(out, self.stddev)

    def __call__(self, latent_state: TransitionLatentState) -> torch.distributions.Normal:
        return super().__call__(latent_state)
