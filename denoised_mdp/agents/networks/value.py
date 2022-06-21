from typing import *

import torch
from torch import jit, nn

from .transition import LatentState as TransitionLatentState
from .utils import BottledModule, get_activation_module


class ValueModel(jit.ScriptModule):
    __constants__ = ['action_size']

    action_size: Optional[int]

    def __init__(self, belief_size, state_size, hidden_size, activation_function="relu",
                 action_size: Optional[int] = None):  # if action_size is not None, Q fn
        super().__init__()
        self.action_size = action_size
        self.net = BottledModule(nn.Sequential(
            nn.Linear(
                belief_size + state_size + (
                    0 if action_size is None else int(action_size)
                ),
                hidden_size,
            ),
            get_activation_module(activation_function),
            nn.Linear(hidden_size, hidden_size),
            get_activation_module(activation_function),
            nn.Linear(hidden_size, hidden_size),
            get_activation_module(activation_function),
            nn.Linear(hidden_size, 1),
        ))

    @jit.script_method
    def forward_feature(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)

    def forward(self, input: Union[torch.Tensor, TransitionLatentState], action: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(input, TransitionLatentState):
            feature = input.x_feature
        else:
            feature = input  # e.g., some imagined latent
        assert (action is None) == (self.action_size is None)
        if action is not None:
           feature = torch.cat([feature, action], dim=-1)
        return self.forward_feature(feature).squeeze(dim=-1)

    def __call__(self, input: Union[torch.Tensor, TransitionLatentState], action: Optional[torch.Tensor] = None) -> torch.Tensor:
        return super().__call__(input, action)

    def extra_repr(self) -> str:
        return f"action_size={self.action_size}"
