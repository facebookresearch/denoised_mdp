# Copyright (c) 2019 Kai Arulkumaran (Original PlaNet parts) Copyright (c) 2020 Yusuke Urakami (Dreamer parts)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import dataclasses
import enum

import torch
from torch import jit, nn
from torch.nn import functional as F
import torch.distributions

from .utils import BottledModule, get_activation_module, no_jit_fuser
from ... import utils


class LatentStateTuple(NamedTuple):
    # This hack is needed because JIT doesn't support dataclasses w/
    # cached computed property or even custom methods.
    # https://github.com/pytorch/pytorch/issues/48984
    x_belief: torch.Tensor
    x_state: torch.Tensor
    y_belief: torch.Tensor
    y_state: torch.Tensor
    z_belief: torch.Tensor
    z_state: torch.Tensor


@dataclasses.dataclass
class LatentState:
    x_belief: torch.Tensor
    x_state: torch.Tensor
    y_belief: torch.Tensor
    y_state: torch.Tensor
    z_belief: torch.Tensor
    z_state: torch.Tensor

    @classmethod
    def stack(cls, list: 'List[LatentState]', dim=0):
        return LatentState(
            x_belief=torch.stack([ls.x_belief for ls in list], dim=dim),
            x_state=torch.stack([ls.x_state for ls in list], dim=dim),
            y_belief=torch.stack([ls.y_belief for ls in list], dim=dim),
            y_state=torch.stack([ls.y_state for ls in list], dim=dim),
            z_belief=torch.stack([ls.z_belief for ls in list], dim=dim),
            z_state=torch.stack([ls.z_state for ls in list], dim=dim),
        )

    def replace(self,
                x: Optional['LatentState'] = None,
                y: Optional['LatentState'] = None,
                z: Optional['LatentState'] = None) -> 'LatentState':
        replace_kwargs = {}
        if x is not None:
            replace_kwargs.update(
                x_belief=x.x_belief,
                x_state=x.x_state,
            )
        if y is not None:
            replace_kwargs.update(
                y_belief=y.y_belief,
                y_state=y.y_state,
            )
        if z is not None:
            replace_kwargs.update(
                z_belief=z.z_belief,
                z_state=z.z_state,
            )
        return dataclasses.replace(self, **replace_kwargs)

    def new_zeros(self, x=False, y=False, z=False) -> 'LatentState':
        return LatentState(
            x_belief=(self.x_belief if not x else torch.zeros_like(self.x_belief)),
            x_state=(self.x_state if not x else torch.zeros_like(self.x_state)),
            y_belief=(self.y_belief if not y else torch.zeros_like(self.y_belief)),
            y_state=(self.y_state if not y else torch.zeros_like(self.y_state)),
            z_belief=(self.z_belief if not z else torch.zeros_like(self.z_belief)),
            z_state=(self.z_state if not z else torch.zeros_like(self.z_state)),
        )

    def new_emptydim(self, x=False, y=False, z=False) -> 'LatentState':
        empty_tensor = self.x_belief.narrow(-1, 0, 0)
        return LatentState(
            x_belief=(self.x_belief if not x else empty_tensor),
            x_state=(self.x_state if not x else empty_tensor),
            y_belief=(self.y_belief if not y else empty_tensor),
            y_state=(self.y_state if not y else empty_tensor),
            z_belief=(self.z_belief if not z else empty_tensor),
            z_state=(self.z_state if not z else empty_tensor),
        )

    @property
    def batch_shape(self) -> torch.Size:
        return self.x_belief.shape[:-1]

    @utils.lazy_property
    def full_feature(self) -> torch.Tensor:
        return torch.cat(
            [
                self.x_belief, self.x_state,
                self.y_belief, self.y_state,
                self.z_belief, self.z_state,
            ], dim=-1)

    @utils.lazy_property
    def x_feature(self) -> torch.Tensor:
        return self.full_feature.narrow(-1, 0, self.x_belief.shape[-1] + self.x_state.shape[-1])

    @utils.lazy_property
    def y_feature(self) -> torch.Tensor:
        x_feature_dim = self.x_belief.shape[-1] + self.x_state.shape[-1]
        y_feature_dim = self.y_belief.shape[-1] + self.y_state.shape[-1]
        return self.full_feature.narrow(-1, x_feature_dim, y_feature_dim)

    def as_namedtuple(self) -> LatentStateTuple:
        return LatentStateTuple(
            x_belief=self.x_belief,
            x_state=self.x_state,
            y_belief=self.y_belief,
            y_state=self.y_state,
            z_belief=self.z_belief,
            z_state=self.z_state,
        )

    def detach(self) -> 'LatentState':
        return LatentState(
            x_belief=self.x_belief.detach(),
            x_state=self.x_state.detach(),
            y_belief=self.y_belief.detach(),
            y_state=self.y_state.detach(),
            z_belief=self.z_belief.detach(),
            z_state=self.z_state.detach(),
        )

    def flatten(self, start_dim=0, end_dim=-2) -> 'LatentState':
        return LatentState(
            x_belief=self.x_belief.flatten(start_dim, end_dim),
            x_state=self.x_state.flatten(start_dim, end_dim),
            y_belief=self.y_belief.flatten(start_dim, end_dim),
            y_state=self.y_state.flatten(start_dim, end_dim),
            z_belief=self.z_belief.flatten(start_dim, end_dim),
            z_state=self.z_state.flatten(start_dim, end_dim),
        )

    def unflatten(self, dim, sizes) -> 'LatentState':
        assert 0 <= dim < len(self.batch_shape)
        return LatentState(
            x_belief=self.x_belief.unflatten(dim, sizes),
            x_state=self.x_state.unflatten(dim, sizes),
            y_belief=self.y_belief.unflatten(dim, sizes),
            y_state=self.y_state.unflatten(dim, sizes),
            z_belief=self.z_belief.unflatten(dim, sizes),
            z_state=self.z_state.unflatten(dim, sizes),
        )

    def narrow(self, dim: int, start: int, length: int) -> 'LatentState':
        assert 0 <= dim < len(self.batch_shape)
        return LatentState(
            x_belief=self.x_belief.narrow(dim, start, length),
            x_state=self.x_state.narrow(dim, start, length),
            y_belief=self.y_belief.narrow(dim, start, length),
            y_state=self.y_state.narrow(dim, start, length),
            z_belief=self.z_belief.narrow(dim, start, length),
            z_state=self.z_state.narrow(dim, start, length),
        )

    def __getitem__(self, slice) -> 'LatentState':
        x_belief = self.x_belief[slice]
        assert x_belief.shape[-1] == self.x_belief.shape[-1], "can only slice batch dims"
        return LatentState(
            x_belief=x_belief,
            x_state=self.x_state[slice],
            y_belief=self.y_belief[slice],
            y_state=self.y_state[slice],
            z_belief=self.z_belief[slice],
            z_state=self.z_state[slice],
        )


class PartialOutputWithoutPosterior(NamedTuple):
    belief: torch.Tensor
    prior_state: torch.Tensor
    prior_noise: torch.Tensor
    prior_mean: torch.Tensor
    prior_stddev: torch.Tensor


class PartialOutputWithPosterior(NamedTuple):
    belief: torch.Tensor
    prior_state: torch.Tensor
    prior_noise: torch.Tensor
    prior_mean: torch.Tensor
    prior_stddev: torch.Tensor
    posterior_state: torch.Tensor
    posterior_noise: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_stddev: torch.Tensor


class LatentPart(enum.Enum):
    x = enum.auto()
    y = enum.auto()
    z = enum.auto()


XYZContrainedT = TypeVar('XYZContrainedT')


@dataclasses.dataclass(frozen=True)
class XYZContainer(Generic[XYZContrainedT]):
    x: XYZContrainedT
    y: XYZContrainedT
    z: XYZContrainedT

    def __getitem__(self, ii: Any) -> XYZContrainedT:
        if ii == 'x' or ii is LatentPart.x:
            return self.x
        elif ii == 'y' or ii is LatentPart.y:
            return self.y
        elif ii == 'z' or ii is LatentPart.z:
            return self.z
        raise ValueError(f"Unexpected index {repr(ii)}")


PartialOutputT = TypeVar('PartialOutputT', PartialOutputWithoutPosterior, PartialOutputWithPosterior)


@dataclasses.dataclass
class _Output(Generic[PartialOutputT]):
    x: PartialOutputT
    y: PartialOutputT
    z: PartialOutputT
    belief: XYZContainer[torch.Tensor] = dataclasses.field(init=False)
    prior_state: XYZContainer[torch.Tensor] = dataclasses.field(init=False)
    prior_noise: XYZContainer[torch.Tensor] = dataclasses.field(init=False)
    prior: XYZContainer[torch.distributions.Normal] = dataclasses.field(init=False)

    def __post_init__(self):
        self.belief = XYZContainer(
            x=self.x.belief, y=self.y.belief, z=self.z.belief,
        )
        self.prior_state = XYZContainer(
            x=self.x.prior_state, y=self.y.prior_state, z=self.z.prior_state,
        )
        self.prior_noise = XYZContainer(
            x=self.x.prior_noise, y=self.y.prior_noise, z=self.z.prior_noise,
        )
        self.prior = XYZContainer(
            x=torch.distributions.Normal(self.x.prior_mean, self.x.prior_stddev),
            y=torch.distributions.Normal(self.y.prior_mean, self.y.prior_stddev),
            z=torch.distributions.Normal(self.z.prior_mean, self.z.prior_stddev),
        )

    @utils.lazy_property
    def prior_latent_state(self) -> LatentState:
        return LatentState(
            x_belief=self.belief.x,
            y_belief=self.belief.y,
            z_belief=self.belief.z,
            x_state=self.prior_state.x,
            y_state=self.prior_state.y,
            z_state=self.prior_state.z,
        )


@dataclasses.dataclass
class OutputWithoutPosterior(_Output[PartialOutputWithoutPosterior]):
    pass


@dataclasses.dataclass
class OutputWithPosterior(_Output[PartialOutputWithPosterior]):
    posterior_state: XYZContainer[torch.Tensor] = dataclasses.field(init=False)
    posterior_noise: XYZContainer[torch.Tensor] = dataclasses.field(init=False)
    posterior: XYZContainer[torch.distributions.Normal] = dataclasses.field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.posterior_state = XYZContainer(
            x=self.x.posterior_state, y=self.y.posterior_state, z=self.z.posterior_state,
        )
        self.posterior_noise = XYZContainer(
            x=self.x.posterior_noise, y=self.y.posterior_noise, z=self.z.posterior_noise,
        )
        self.posterior = XYZContainer(
            x=torch.distributions.Normal(self.x.posterior_mean, self.x.posterior_stddev),
            y=torch.distributions.Normal(self.y.posterior_mean, self.y.posterior_stddev),
            z=torch.distributions.Normal(self.z.posterior_mean, self.z.posterior_stddev),
        )

    @utils.lazy_property
    def posterior_latent_state(self) -> LatentState:
        return LatentState(
            x_belief=self.belief.x,
            y_belief=self.belief.y,
            z_belief=self.belief.z,
            x_state=self.posterior_state.x,
            y_state=self.posterior_state.y,
            z_state=self.posterior_state.z,
        )


class DummyEmptyGRUCell(nn.Module):
    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        assert input.shape[-1] == hidden.shape[-1] == 0
        return hidden


class DummyEmptyNet(nn.Module):
    def forward(self, input: torch.Tensor):
        assert input.shape[-1] == 0
        return input


class TransitionModel(jit.ScriptModule):
    __constants__ = [
        "min_stddev", "action_size",
        "x_belief_size", "x_state_size",
        "y_belief_size", "y_state_size",
        "z_belief_size", "z_state_size",
        "embedding_size",
    ]

    x_belief_size: int
    x_state_size: int
    y_belief_size: int
    y_state_size: int
    z_belief_size: int
    z_state_size: int
    embedding_size: int
    action_size: int

    @property
    def only_x(self) -> bool:
        return self.y_belief_size == self.z_belief_size == 0

    def __init__(
        self,
        x_belief_size: int,  # h_t
        x_state_size: int,   # s_t
        y_belief_size: int,  # h_t
        y_state_size: int,   # s_t
        z_belief_size: int,  # h_t
        z_state_size: int,   # s_t
        action_size: int,
        hidden_size: int,
        embedding_size: int,  # enc(o_t)
        activation_function: str = "relu",
        min_stddev: float = 0.1,
    ):
        super().__init__()
        self.min_stddev = min_stddev
        self.x_belief_size = x_belief_size
        self.x_state_size = x_state_size
        self.y_belief_size = y_belief_size
        self.y_state_size = y_state_size
        self.z_belief_size = z_belief_size
        self.z_state_size = z_state_size
        self.embedding_size = embedding_size
        self.action_size = action_size

        assert (x_belief_size > 0) and (x_state_size > 0)
        assert (y_belief_size > 0) == (y_state_size > 0)
        assert (z_belief_size > 0) == (z_state_size > 0)

        # x
        self.x_state_action_pre_rnn = BottledModule(nn.Sequential(
            nn.Linear(x_state_size + action_size, x_belief_size),
            get_activation_module(activation_function),
        ))
        self.x_rnn = nn.GRUCell(x_belief_size, x_belief_size)

        self.x_belief_to_state_prior = BottledModule(nn.Sequential(
            nn.Linear(x_belief_size, x_belief_size),
            get_activation_module(activation_function),
            nn.Linear(x_belief_size, 2 * x_state_size),
        ))

        # y
        if y_belief_size > 0:
            self.y_state_pre_rnn = BottledModule(nn.Sequential(
                nn.Linear(y_state_size, y_belief_size),
                get_activation_module(activation_function),
            ))
            self.y_rnn = nn.GRUCell(y_belief_size, y_belief_size)
            self.y_belief_to_state_prior = BottledModule(nn.Sequential(
                nn.Linear(y_belief_size, y_belief_size),
                get_activation_module(activation_function),
                nn.Linear(y_belief_size, 2 * y_state_size),
            ))
        else:
            self.y_state_pre_rnn = BottledModule(DummyEmptyNet())
            self.y_rnn = DummyEmptyGRUCell()
            self.y_belief_to_state_prior = BottledModule(DummyEmptyNet())

        # z
        if z_belief_size > 0:
            self.z_state_action_x_pre_rnn = BottledModule(nn.Sequential(
                nn.Linear(
                    x_belief_size + x_state_size + 
                    y_belief_size + y_state_size + 
                    z_state_size + action_size,
                    z_belief_size,
                ),
                get_activation_module(activation_function),
                nn.Linear(z_belief_size, z_belief_size),
                get_activation_module(activation_function),
            ))
            self.z_rnn = nn.GRUCell(z_belief_size, z_belief_size)
            self.z_belief_to_state_prior = BottledModule(nn.Sequential(
                nn.Linear(z_belief_size, z_belief_size),
                get_activation_module(activation_function),
                nn.Linear(z_belief_size, 2 * z_state_size),
            ))
        else:
            self.z_state_action_x_pre_rnn = BottledModule(DummyEmptyNet())
            self.z_rnn = DummyEmptyGRUCell()
            self.z_belief_to_state_prior = BottledModule(DummyEmptyNet())

        # posterior

        self.xy_belief_obs_to_state_posterior = BottledModule(nn.Sequential(
            nn.Linear(
                x_belief_size + y_belief_size + z_belief_size + z_state_size + embedding_size,
                x_belief_size + y_belief_size,
            ),
            get_activation_module(activation_function),
            nn.Linear(
                x_belief_size + y_belief_size,
                2 * x_state_size + 2 * y_state_size,
            ),
        ))

        if z_belief_size > 0:
            self.z_belief_obs_to_state_posterior = BottledModule(nn.Sequential(
                nn.Linear(
                    x_belief_size + y_belief_size + z_belief_size + embedding_size + x_state_size + y_state_size,
                    z_belief_size,
                ),
                get_activation_module(activation_function),
                nn.Linear(z_belief_size, 2 * z_state_size),
            ))
        else:
            self.z_belief_obs_to_state_posterior = BottledModule(DummyEmptyNet())

    def init_latent_state(self, *, batch_shape: Tuple[int, ...] = ()) -> LatentState:
        zero = self.x_rnn.weight_ih.new_zeros(())
        return LatentState(
            x_belief=zero.expand(*batch_shape, self.x_belief_size),
            x_state=zero.expand(*batch_shape, self.x_state_size),
            y_belief=zero.expand(*batch_shape, self.y_belief_size),
            y_state=zero.expand(*batch_shape, self.y_state_size),
            z_belief=zero.expand(*batch_shape, self.z_belief_size),
            z_state=zero.expand(*batch_shape, self.z_state_size),
        )

    # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :    0    1    2    3    4    5
    # o :        -X--X--X--X--X-
    # a : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    @jit.script_method
    def forward_generic(
        self,
        previous_latent_state: LatentStateTuple,
        actions: torch.Tensor,
        next_observations: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        next_observation_nonfirststeps: Optional[torch.Tensor] = None,  # whether `next_observation` is the beginning of an episode
    ) -> Tuple[PartialOutputWithPosterior, PartialOutputWithPosterior, PartialOutputWithPosterior]:
        """
        Input: init_belief, init_state:    torch.Size([50, 200]) torch.Size([50, 30])
        Output: beliefs, prior_states, prior_means, prior_stddevs, posterior_states, posterior_means, posterior_stddevs
                        torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        """
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        assert actions.size(-1) == self.action_size
        T: int = actions.size(0)
        assert T > 0
        empty_tensor = actions[0].narrow(-1, 0, 0)
        (
            x_beliefs,
            x_prior_states,
            x_prior_means,
            x_prior_stddevs,
            x_posterior_states,
            x_posterior_means,
            x_posterior_stddevs,
            y_beliefs,
            y_prior_states,
            y_prior_means,
            y_prior_stddevs,
            y_posterior_states,
            y_posterior_means,
            y_posterior_stddevs,
            z_beliefs,
            z_prior_states,
            z_prior_means,
            z_prior_stddevs,
            z_posterior_states,
            z_posterior_means,
            z_posterior_stddevs,
        ) = (
            [empty_tensor] * T,
            [empty_tensor] * T,
            [empty_tensor] * T,
            [empty_tensor] * T,
            [empty_tensor] * T,
            [empty_tensor] * T,
            [empty_tensor] * T,
            [empty_tensor] * T,
            [empty_tensor] * T,
            [empty_tensor] * T,
            [empty_tensor] * T,
            [empty_tensor] * T,
            [empty_tensor] * T,
            [empty_tensor] * T,
            [empty_tensor] * T,
            [empty_tensor] * T,
            [empty_tensor] * T,
            [empty_tensor] * T,
            [empty_tensor] * T,
            [empty_tensor] * T,
            [empty_tensor] * T,
        )
        prev_x_belief = previous_latent_state.x_belief
        prev_x_prior_state = previous_latent_state.x_state
        prev_x_posterior_state = previous_latent_state.x_state
        prev_y_belief = previous_latent_state.y_belief
        prev_y_prior_state = previous_latent_state.y_state
        prev_y_posterior_state = previous_latent_state.y_state
        prev_z_belief = previous_latent_state.z_belief
        prev_z_prior_state = previous_latent_state.z_state
        prev_z_posterior_state = previous_latent_state.z_state

        assert int(next_observations is None) == int(rewards is None)

        x_prior_noises = torch.randn(
            list(actions.shape[:-1]) + [self.x_state_size],
            dtype=actions.dtype, device=actions.device)
        y_prior_noises = torch.randn(
            list(actions.shape[:-1]) + [self.y_state_size],
            dtype=actions.dtype, device=actions.device)
        z_prior_noises = torch.randn(
            list(actions.shape[:-1]) + [self.z_state_size],
            dtype=actions.dtype, device=actions.device)

        if next_observations is not None:
            x_posterior_noises = torch.randn_like(x_prior_noises)
            y_posterior_noises = torch.randn_like(y_prior_noises)
            z_posterior_noises = torch.randn_like(z_prior_noises)
        else:
            x_posterior_noises = torch.randn([T, 0], dtype=actions.dtype, device=actions.device)
            y_posterior_noises = torch.randn([T, 0], dtype=actions.dtype, device=actions.device)
            z_posterior_noises = torch.randn([T, 0], dtype=actions.dtype, device=actions.device)

        # Loop over time sequence
        for t in range(T):
            action = actions[t]
            # Select appropriate previous state
            if next_observations is None:
                prev_x_state = prev_x_prior_state
                prev_y_state = prev_y_prior_state
                prev_z_state = prev_z_prior_state
            else:
                prev_x_state = prev_x_posterior_state
                prev_y_state = prev_y_posterior_state
                prev_z_state = prev_z_posterior_state

            # Mask if previous transition was terminal
            if next_observation_nonfirststeps is not None:
                next_observation_nonfirststep = next_observation_nonfirststeps[t].unsqueeze(-1)
                prev_x_belief = prev_x_belief * next_observation_nonfirststep
                prev_x_state = prev_x_state * next_observation_nonfirststep
                prev_y_belief = prev_y_belief * next_observation_nonfirststep
                prev_y_state = prev_y_state * next_observation_nonfirststep
                prev_z_belief = prev_z_belief * next_observation_nonfirststep
                prev_z_state = prev_z_state * next_observation_nonfirststep
                action = action * next_observation_nonfirststep

            # [X prior] Compute belief (deterministic hidden state)
            prev_x_state_action = torch.cat([prev_x_state, action], dim=-1)
            x_belief = self.x_rnn(
                self.x_state_action_pre_rnn(prev_x_state_action),
                prev_x_belief,
            )
            # [X prior] Compute state prior by applying transition dynamics
            x_prior_mean, _x_prior_stddev = torch.chunk(self.x_belief_to_state_prior(x_belief), 2, dim=-1)
            x_prior_stddev = F.softplus(_x_prior_stddev) + self.min_stddev
            x_prior_state = x_prior_mean + x_prior_stddev * x_prior_noises[t]
            # [X prior] save results
            x_beliefs[t] = x_belief
            x_prior_means[t] = x_prior_mean
            x_prior_stddevs[t] = x_prior_stddev
            x_prior_states[t] = x_prior_state

            if self.y_belief_size > 0:  # JIT doesn't seem to like empty tensors that much, so use this
                # [Y prior] Compute belief (deterministic hidden state)
                y_belief = self.y_rnn(
                    self.y_state_pre_rnn(prev_y_state),
                    prev_y_belief,
                )
                # [Y prior] Compute state prior by applying transition dynamics
                y_prior_mean, _y_prior_stddev = torch.chunk(self.y_belief_to_state_prior(y_belief), 2, dim=-1)
                y_prior_stddev = F.softplus(_y_prior_stddev) + self.min_stddev
                y_prior_state = y_prior_mean + y_prior_stddev * y_prior_noises[t]
                # [Y prior] save results
                y_beliefs[t] = y_belief
                y_prior_means[t] = y_prior_mean
                y_prior_stddevs[t] = y_prior_stddev
                y_prior_states[t] = y_prior_state
            else:
                y_belief = y_beliefs[t]

            # [XY posterior]
            if next_observations is not None:
                # Compute state posterior by applying transition dynamics (i.e., using *_belief) and using current observation
                xy_posterior_input = torch.cat(
                    [x_belief, y_belief, prev_z_belief, prev_z_state, next_observations[t]],
                    dim=-1,
                )
                xy_posterior_mean, _xy_posterior_stddev = torch.chunk(
                    self.xy_belief_obs_to_state_posterior(xy_posterior_input), 2,
                    dim=-1,
                )

                if self.y_belief_size == 0:  # somehow =0 messes up with JIT...
                    x_posterior_mean = xy_posterior_mean
                    x_posterior_stddev: torch.Tensor = F.softplus(_xy_posterior_stddev) + self.min_stddev
                    y_posterior_mean = empty_tensor
                    y_posterior_stddev = empty_tensor
                else:
                    x_posterior_mean, y_posterior_mean = xy_posterior_mean.split([self.x_state_size, self.y_state_size], dim=-1)
                    xy_posterior_stddev: torch.Tensor = F.softplus(_xy_posterior_stddev) + self.min_stddev
                    x_posterior_stddev, y_posterior_stddev = xy_posterior_stddev.split([self.x_state_size, self.y_state_size], dim=-1)

                x_posterior_state = x_posterior_mean + x_posterior_stddev * x_posterior_noises[t]
                y_posterior_state = y_posterior_mean + y_posterior_stddev * y_posterior_noises[t]

                x_posterior_means[t] = x_posterior_mean
                x_posterior_stddevs[t] = x_posterior_stddev
                x_posterior_states[t] = x_posterior_state
                y_posterior_means[t] = y_posterior_mean
                y_posterior_stddevs[t] = y_posterior_stddev
                y_posterior_states[t] = y_posterior_state

                x_state_for_z_belief = x_posterior_state
                y_state_for_z_belief = y_posterior_state
            else:
                x_posterior_state = x_posterior_states[t]
                y_posterior_state = y_posterior_states[t]
                x_state_for_z_belief = x_prior_state
                y_state_for_z_belief = y_prior_state

            if self.z_belief_size > 0:
                # [Z prior] Compute belief (deterministic hidden state)
                z_belief = self.z_rnn(
                    self.z_state_action_x_pre_rnn(
                        torch.cat(
                            [
                                x_belief,
                                x_state_for_z_belief,
                                y_belief,
                                y_state_for_z_belief,
                                prev_z_state,
                                action,
                            ],
                            dim=-1,
                        ),
                    ),
                    prev_z_belief,
                )
                # [Z prior] Compute state prior by applying transition dynamics
                z_prior_mean, _z_prior_stddev = torch.chunk(self.z_belief_to_state_prior(z_belief), 2, dim=-1)
                z_prior_stddev = F.softplus(_z_prior_stddev) + self.min_stddev
                z_prior_state = z_prior_mean + z_prior_stddev * z_prior_noises[t]
                # [Z prior] save results
                z_beliefs[t] = z_belief
                z_prior_means[t] = z_prior_mean
                z_prior_stddevs[t] = z_prior_stddev
                z_prior_states[t] = z_prior_state

                # [Z posterior]
                if next_observations is not None:
                    # Compute state posterior by applying transition dynamics (i.e., using *_belief) and using current observation
                    z_posterior_input = torch.cat(
                        [x_belief, x_posterior_state, y_belief, y_posterior_state, z_belief, next_observations[t]],
                        dim=-1,
                    )
                    z_posterior_mean, _z_posterior_stddev = torch.chunk(
                        self.z_belief_obs_to_state_posterior(z_posterior_input), 2,
                        dim=-1,
                    )
                    z_posterior_stddev = F.softplus(_z_posterior_stddev) + self.min_stddev
                    z_posterior_state = z_posterior_mean + z_posterior_stddev * z_posterior_noises[t]

                    z_posterior_means[t] = z_posterior_mean
                    z_posterior_stddevs[t] = z_posterior_stddev
                    z_posterior_states[t] = z_posterior_state

            prev_x_belief = x_beliefs[t]
            prev_y_belief = y_beliefs[t]
            prev_z_belief = z_beliefs[t]
            prev_x_prior_state = x_prior_states[t]
            prev_y_prior_state = y_prior_states[t]
            prev_z_prior_state = z_prior_states[t]
            prev_x_posterior_state = x_posterior_states[t]
            prev_y_posterior_state = y_posterior_states[t]
            prev_z_posterior_state = z_posterior_states[t]

        # Return new hidden states
        return (
            PartialOutputWithPosterior(
                belief=torch.stack(x_beliefs, dim=0),
                prior_state=torch.stack(x_prior_states, dim=0),
                prior_noise=x_prior_noises,
                prior_mean=torch.stack(x_prior_means, dim=0),
                prior_stddev=torch.stack(x_prior_stddevs, dim=0),
                posterior_state=torch.stack(x_posterior_states, dim=0),
                posterior_noise=x_posterior_noises,
                posterior_mean=torch.stack(x_posterior_means, dim=0),
                posterior_stddev=torch.stack(x_posterior_stddevs, dim=0),
            ),
            PartialOutputWithPosterior(
                belief=torch.stack(y_beliefs, dim=0),
                prior_state=torch.stack(y_prior_states, dim=0),
                prior_noise=y_prior_noises,
                prior_mean=torch.stack(y_prior_means, dim=0),
                prior_stddev=torch.stack(y_prior_stddevs, dim=0),
                posterior_state=torch.stack(y_posterior_states, dim=0),
                posterior_noise=y_posterior_noises,
                posterior_mean=torch.stack(y_posterior_means, dim=0),
                posterior_stddev=torch.stack(y_posterior_stddevs, dim=0),
            ),
            PartialOutputWithPosterior(
                belief=torch.stack(z_beliefs, dim=0),
                prior_state=torch.stack(z_prior_states, dim=0),
                prior_noise=z_prior_noises,
                prior_mean=torch.stack(z_prior_means, dim=0),
                prior_stddev=torch.stack(z_prior_stddevs, dim=0),
                posterior_state=torch.stack(z_posterior_states, dim=0),
                posterior_noise=z_posterior_noises,
                posterior_mean=torch.stack(z_posterior_means, dim=0),
                posterior_stddev=torch.stack(z_posterior_stddevs, dim=0),
            ),
        )

    def posterior_rsample(
        self,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor, *,
        next_observation_nonfirststeps: Optional[torch.Tensor] = None,
        previous_latent_state: Optional[LatentState] = None,
    ) -> OutputWithPosterior:
        assert actions.ndim == 3 and actions.shape[-1] == self.action_size, "must have [T, B, *] shape"
        if previous_latent_state is None:
            previous_latent_state = self.init_latent_state(batch_shape=(actions.shape[1],))
        else:
            assert len(previous_latent_state.batch_shape) == 2, "must have [T, B, *] shape"
        with no_jit_fuser():  # https://github.com/pytorch/pytorch/issues/68800
            x, y, z = self.forward_generic(
                previous_latent_state.as_namedtuple(), actions, next_observations,
                rewards, next_observation_nonfirststeps)
        return OutputWithPosterior(x, y, z)

    @jit.script_method
    def forward_generic_one_step(
        self,
        previous_latent_state: LatentStateTuple,
        action: torch.Tensor,
        next_observation: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
        next_observation_nonfirststep: Optional[torch.Tensor] = None,
    ) -> Tuple[PartialOutputWithPosterior, PartialOutputWithPosterior, PartialOutputWithPosterior]:
        assert action.size(-1) == self.action_size
        assert int(next_observation is None) == int(reward is None)

        prev_x_belief = previous_latent_state.x_belief
        prev_x_state = previous_latent_state.x_state
        prev_y_belief = previous_latent_state.y_belief
        prev_y_state = previous_latent_state.y_state
        prev_z_belief = previous_latent_state.z_belief
        prev_z_state = previous_latent_state.z_state

        x_prior_noise = torch.randn_like(prev_x_state)
        y_prior_noise = torch.randn_like(prev_y_state)
        z_prior_noise = torch.randn_like(prev_z_state)

        empty_tensor = action.narrow(-1, 0, 0)
        if next_observation is not None:
            x_posterior_noise = torch.randn_like(x_prior_noise)
            y_posterior_noise = torch.randn_like(y_prior_noise)
            z_posterior_noise = torch.randn_like(z_prior_noise)
        else:
            x_posterior_noise = empty_tensor
            y_posterior_noise = empty_tensor
            z_posterior_noise = empty_tensor

        # Mask if previous transition was terminal
        if next_observation_nonfirststep is not None:
            next_observation_nonfirststep = next_observation_nonfirststep.unsqueeze(-1)
            prev_x_belief = prev_x_belief * next_observation_nonfirststep
            prev_x_state = prev_x_state * next_observation_nonfirststep
            prev_y_belief = prev_y_belief * next_observation_nonfirststep
            prev_y_state = prev_y_state * next_observation_nonfirststep
            prev_z_belief = prev_z_belief * next_observation_nonfirststep
            prev_z_state = prev_z_state * next_observation_nonfirststep
            action = action * next_observation_nonfirststep

        # [X prior] Compute belief (deterministic hidden state)
        prev_x_state_action = torch.cat([prev_x_state, action], dim=-1)
        x_belief = self.x_rnn(
            self.x_state_action_pre_rnn(prev_x_state_action),
            prev_x_belief,
        )
        # [X prior] Compute state prior by applying transition dynamics
        x_prior_mean, _x_prior_stddev = torch.chunk(self.x_belief_to_state_prior(x_belief), 2, dim=-1)
        x_prior_stddev = F.softplus(_x_prior_stddev) + self.min_stddev
        x_prior_state = x_prior_mean + x_prior_stddev * x_prior_noise

        if self.y_belief_size > 0:  # JIT doesn't seem to like empty tensors that much, so use this
            # [Y prior] Compute belief (deterministic hidden state)
            y_belief = self.y_rnn(
                self.y_state_pre_rnn(prev_y_state),
                prev_y_belief,
            )
            # [Y prior] Compute state prior by applying transition dynamics
            y_prior_mean, _y_prior_stddev = torch.chunk(self.y_belief_to_state_prior(y_belief), 2, dim=-1)
            y_prior_stddev = F.softplus(_y_prior_stddev) + self.min_stddev
            y_prior_state = y_prior_mean + y_prior_stddev * y_prior_noise
        else:
            y_belief = empty_tensor
            y_prior_mean = empty_tensor
            y_prior_stddev = empty_tensor
            y_prior_state = empty_tensor

        # [XY posterior]
        if next_observation is not None:
            # Compute state posterior by applying transition dynamics and using current observation
            xy_posterior_input = torch.cat(
                [x_belief, y_belief, prev_z_belief, prev_z_state, next_observation],
                dim=-1,
            )

            xy_posterior_mean, _xy_posterior_stddev = torch.chunk(
                self.xy_belief_obs_to_state_posterior(xy_posterior_input), 2,
                dim=-1,
            )
            if self.y_belief_size == 0:  # somehow =0 messes up with JIT...
                x_posterior_mean = xy_posterior_mean
                x_posterior_stddev: torch.Tensor = F.softplus(_xy_posterior_stddev) + self.min_stddev
                y_posterior_mean = empty_tensor
                y_posterior_stddev = empty_tensor
            else:
                x_posterior_mean, y_posterior_mean = xy_posterior_mean.split([self.x_state_size, self.y_state_size], dim=-1)
                xy_posterior_stddev: torch.Tensor = F.softplus(_xy_posterior_stddev) + self.min_stddev
                x_posterior_stddev, y_posterior_stddev = xy_posterior_stddev.split([self.x_state_size, self.y_state_size], dim=-1)

            y_posterior_state = y_posterior_mean + y_posterior_stddev * y_posterior_noise
            x_posterior_state = x_posterior_mean + x_posterior_stddev * x_posterior_noise

            x_state_for_z_belief = x_posterior_state
        else:
            x_posterior_mean = empty_tensor
            y_posterior_mean = empty_tensor
            x_posterior_stddev = empty_tensor
            y_posterior_stddev = empty_tensor
            x_posterior_state = empty_tensor
            y_posterior_state = empty_tensor
            x_state_for_z_belief = x_prior_state

        z_posterior_mean = empty_tensor
        z_posterior_stddev = empty_tensor
        z_posterior_state = empty_tensor
        if self.z_belief_size > 0:
            # [Z prior] Compute belief (deterministic hidden state)
            z_belief = self.z_rnn(
                self.z_state_action_x_pre_rnn(
                    torch.cat(
                        [
                            x_belief,
                            x_state_for_z_belief,
                            prev_z_state,
                            action,
                        ],
                        dim=-1,
                    ),
                ),
                prev_z_belief,
            )
            # [Z prior] Compute state prior by applying transition dynamics
            z_prior_mean, _z_prior_stddev = torch.chunk(self.z_belief_to_state_prior(z_belief), 2, dim=-1)
            z_prior_stddev = F.softplus(_z_prior_stddev) + self.min_stddev
            z_prior_state = z_prior_mean + z_prior_stddev * z_prior_noise

            # [Z posterior]
            if next_observation is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                z_posterior_input = torch.cat(
                    [x_belief, x_posterior_state, y_belief, y_posterior_state, z_belief, next_observation], dim=-1)

                z_posterior_mean, _z_posterior_stddev = torch.chunk(
                    self.z_belief_obs_to_state_posterior(z_posterior_input), 2,
                    dim=-1,
                )
                z_posterior_stddev = F.softplus(_z_posterior_stddev) + self.min_stddev
                z_posterior_state = z_posterior_mean + z_posterior_stddev * z_posterior_noise
        else:
            z_belief = empty_tensor
            z_prior_mean = empty_tensor
            z_prior_stddev = empty_tensor
            z_prior_state = empty_tensor

        # Return new hidden states
        return (
            PartialOutputWithPosterior(
                belief=x_belief,
                prior_state=x_prior_state,
                prior_noise=x_prior_noise,
                prior_mean=x_prior_mean,
                prior_stddev=x_prior_stddev,
                posterior_state=x_posterior_state,
                posterior_noise=x_posterior_noise,
                posterior_mean=x_posterior_mean,
                posterior_stddev=x_posterior_stddev,
            ),
            PartialOutputWithPosterior(
                belief=y_belief,
                prior_state=y_prior_state,
                prior_noise=y_prior_noise,
                prior_mean=y_prior_mean,
                prior_stddev=y_prior_stddev,
                posterior_state=y_posterior_state,
                posterior_noise=y_posterior_noise,
                posterior_mean=y_posterior_mean,
                posterior_stddev=y_posterior_stddev,
            ),
            PartialOutputWithPosterior(
                belief=z_belief,
                prior_state=z_prior_state,
                prior_noise=z_prior_noise,
                prior_mean=z_prior_mean,
                prior_stddev=z_prior_stddev,
                posterior_state=z_posterior_state,
                posterior_noise=z_posterior_noise,
                posterior_mean=z_posterior_mean,
                posterior_stddev=z_posterior_stddev,
            ),
        )

    @jit.script_method
    def forward_x_prior_one_step(
        self,
        previous_latent_state: LatentStateTuple,
        action: torch.Tensor,
    ) -> Tuple[PartialOutputWithoutPosterior, PartialOutputWithoutPosterior, PartialOutputWithoutPosterior]:
        assert action.size(-1) == self.action_size

        prev_x_belief = previous_latent_state.x_belief
        prev_x_state = previous_latent_state.x_state
        x_prior_noise = torch.randn_like(prev_x_state)

        # [X prior] Compute belief (deterministic hidden state)
        prev_x_state_action = torch.cat([prev_x_state, action], dim=-1)
        x_belief = self.x_rnn(
            self.x_state_action_pre_rnn(prev_x_state_action),
            prev_x_belief,
        )
        # [X prior] Compute state prior by applying transition dynamics
        x_prior_mean, _x_prior_stddev = torch.chunk(self.x_belief_to_state_prior(x_belief), 2, dim=-1)
        x_prior_stddev = F.softplus(_x_prior_stddev) + self.min_stddev
        x_prior_state = x_prior_mean + x_prior_stddev * x_prior_noise

        empty_tensor = action.narrow(-1, 0, 0)
        return (
            PartialOutputWithoutPosterior(
                belief=x_belief,
                prior_state=x_prior_state,
                prior_noise=x_prior_noise,
                prior_mean=x_prior_mean,
                prior_stddev=x_prior_stddev,
            ),
            PartialOutputWithoutPosterior(
                belief=empty_tensor,
                prior_state=empty_tensor,
                prior_noise=empty_tensor,
                prior_mean=empty_tensor,
                prior_stddev=empty_tensor,
            ),
            PartialOutputWithoutPosterior(
                belief=empty_tensor,
                prior_state=empty_tensor,
                prior_noise=empty_tensor,
                prior_mean=empty_tensor,
                prior_stddev=empty_tensor,
            ),
        )

    def x_prior_rsample_one_step(
        self,
        action: torch.Tensor, *,
        previous_latent_state: Optional[LatentState] = None,
    ) -> OutputWithoutPosterior:
        assert action.ndim == 2 and action.shape[-1] == self.action_size, "must have [B, *] shape"
        if previous_latent_state is None:
            previous_latent_state = self.init_latent_state(batch_shape=(action.shape[0],))
        else:
            assert len(previous_latent_state.batch_shape) == 1, "must have [B, *] shape"
        with no_jit_fuser():  # https://github.com/pytorch/pytorch/issues/68800
            x, y, z = self.forward_x_prior_one_step(
                previous_latent_state.as_namedtuple(), action)
        return OutputWithoutPosterior(x, y, z)

    def posterior_rsample_one_step(
        self,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        reward: torch.Tensor, *,
        next_observation_nonfirststep: Optional[torch.Tensor] = None,
        previous_latent_state: Optional[LatentState] = None,
    ) -> OutputWithPosterior:
        assert action.ndim == 2 and action.shape[-1] == self.action_size, "must have [B, *] shape"
        if previous_latent_state is None:
            previous_latent_state = self.init_latent_state(batch_shape=(action.shape[0],))
        else:
            assert len(previous_latent_state.batch_shape) == 1, "must have [B, *] shape"
        with no_jit_fuser():  # https://github.com/pytorch/pytorch/issues/68800
            x, y, z = self.forward_generic_one_step(
                previous_latent_state.as_namedtuple(), action,
                next_observation, reward, next_observation_nonfirststep)
        return OutputWithPosterior(x, y, z)
