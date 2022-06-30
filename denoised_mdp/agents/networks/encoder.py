# Copyright (c) 2019 Kai Arulkumaran (Original PlaNet parts) Copyright (c) 2020 Yusuke Urakami (Dreamer parts)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import torch
from torch import jit, nn
import torch.distributions

from .utils import BottledModule, get_activation_module


class EncoderModel(jit.ScriptModule):
    def __init__(self, observation_shape, output_size, activation_function="relu",
                 filter_base=32):
        super().__init__()
        assert len(observation_shape) == 3 and observation_shape[1] == observation_shape[2]
        assert observation_shape[1] in {64, 96}
        conv_part = [
            nn.Conv2d(observation_shape[0], filter_base, 4, stride=2),
            get_activation_module(activation_function),
            nn.Conv2d(filter_base, filter_base * 2, 4, stride=2),
            get_activation_module(activation_function),
            nn.Conv2d(filter_base * 2, filter_base * 4, 4, stride=2),
            get_activation_module(activation_function),
            nn.Conv2d(filter_base * 4, filter_base * 8, 4, stride=2),
            get_activation_module(activation_function),
        ]
        if observation_shape[1] == 64:
            out_dim = filter_base * 8 * 2 * 2
        elif observation_shape[1] == 96:
            conv_part.extend([
                nn.Conv2d(filter_base * 8, filter_base * 8, 3, stride=1),
                get_activation_module(activation_function),
            ])
            out_dim = filter_base * 8 * 2 * 2
        self.net = BottledModule(nn.Sequential(
            *conv_part,
            nn.Flatten(1),
            nn.Linear(out_dim, output_size),
        ), input_elem_dim=3)

    @jit.script_method
    def forward(self, observation):
        return self.net(observation)
