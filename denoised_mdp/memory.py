# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import sys
if sys.version_info < (3, 8):
    from typing_extensions import Protocol

import weakref
import abc

import attrs
import atexit
import tempfile
import os
import shutil
import uuid
import dataclasses
import concurrent.futures
from collections import defaultdict
import logging

from tqdm.auto import tqdm

import numpy as np
import torch

from .envs import AutoResetEnvBase


T = TypeVar('T')


############################################################################
# Simply checksum-like mechanism to check saved trajectory consistency
############################################################################

def checkxor(x: torch.Tensor, *, order=4, reducendim=None):
    if reducendim is None:
        reducendim = 0
    reducendim = reducendim % x.ndim
    xnp = x.detach().cpu().flatten(-reducendim, -1).numpy().view(np.uint8)
    cs = torch.empty(*x.shape[:-reducendim], order, dtype=torch.uint8)
    for ii in range(order):
        skip = int(2 ** ii)
        cs[..., ii] = torch.as_tensor(np.bitwise_xor.reduce(xnp[..., ::skip], -1))
    return cs


def verifyxor(x: torch.Tensor, cs: torch.Tensor):
    order = cs.shape[-1]
    reducendim = x.ndim - (cs.ndim - 1)
    return torch.equal(checkxor(x, order=order, reducendim=reducendim), cs)


############################################################################
# Manager of generic sequential data as Dict[str, torch.Tensor]
# This is used to implement replay buffers
############################################################################

class BaseAccessor(abc.ABC):
    Data_t = Dict[str, torch.Tensor]

    class SamplerCache:
        def __init__(self, accessor: 'BaseAccessor') -> None:
            self.accessor_r = weakref.ref(accessor)
            self.last_updated: DefaultDict[Any, Tuple[int, int]] = defaultdict(lambda: (-1, -1))
            self.cache: Dict[Any, Any] = {}

        def get(self, k, fn: Callable[[], T],
                depends_on_partial_data: bool = False) -> T:
            last_num_compelete_samples, last_num_partial_samples = self.last_updated[k]
            accessor = self.accessor_r()
            assert accessor is not None
            if last_num_compelete_samples != accessor.num_compelete_samples or \
                    (depends_on_partial_data and last_num_partial_samples != accessor.num_partial_samples):
                self.cache[k] = fn()
                self.last_updated[k] = last_num_compelete_samples, last_num_partial_samples
            return self.cache[k]

        def clear_cache(self):
            self.cache.clear()

    def __init__(self, elem_infos: Dict[str, Tuple[torch.Size, torch.dtype]], seed: int, save_dir: str) -> None:
        super().__init__()
        self.elem_infos = elem_infos
        self.np_rng: np.random.Generator = np.random.Generator(np.random.PCG64(seed))
        self.partial_data: Optional['BaseAccessor.Data_t'] = None  # last data can be partial
        # this is larger than num_data to reduce #resize
        self._data_num_samples_storage: torch.Tensor = torch.empty(20, dtype=torch.int64)
        self.num_samples = 0
        self.num_partial_samples = 0
        self.num_data = 0

        self._sampler_cache = BaseAccessor.SamplerCache(self)

        # add a subfolder under save_dir
        save_dir = os.path.join(save_dir, str(uuid.uuid4()))
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self._saver = concurrent.futures.ThreadPoolExecutor(2, initializer=lambda: torch.set_num_threads(1))
        self._saver_outstanding_futures: Dict[concurrent.futures.Future, int] = {}  # outstanding ones

    def __del__(self):
        self._saver.shutdown(True)

    def save_all_complete_data(self, *, move_to_dir: Optional[str] = None):
        for _ in concurrent.futures.as_completed(list(self._saver_outstanding_futures)):
            pass
        torch.save(self.get_complete_data_metadata(), os.path.join(self.save_dir, "metadata.pth"))
        if move_to_dir is not None:
            assert not os.path.exists(move_to_dir)
            shutil.move(self.save_dir, move_to_dir)

    def get_complete_data_metadata(self) -> Dict[str, Any]:
        return dict(
            elem_infos=self.elem_infos,
            np_rng_PCG64_state=self.np_rng.bit_generator.state,
            data_num_samples_storage=self._data_num_samples_storage,
            num_data=self.num_data,
            num_samples=self.num_samples,
        )

    def _prepare_before_load_from_folder(self, metadata):
        self._data_num_samples_storage.resize_as_(metadata['data_num_samples_storage'])

    def _finalize_after_load_from_folder(self, metadata):
        self.np_rng.bit_generator.state = metadata['np_rng_PCG64_state']
        assert self.num_data == metadata['num_data']
        assert self.num_samples == metadata['num_samples']
        assert (self.data_num_samples == cast(torch.Tensor, metadata['data_num_samples_storage'][:self.num_data])).all()
        assert self.partial_data is None
        assert self.num_partial_samples == 0

    def load_from_folder(self, load_dir: str):
        metadata = torch.load(os.path.join(load_dir, "metadata.pth"))
        self._prepare_before_load_from_folder(metadata)

        for data_idx in tqdm(range(metadata['num_data']), desc=f"loading from {load_dir}"):
            complete_data: 'BaseAccessor.Data_t'
            loaded = torch.load(os.path.join(load_dir, f"{data_idx:010d}.pth"), map_location='cpu')
            if isinstance(loaded, tuple):
                check_data: Dict[str, torch.Tensor]
                complete_data, check_data = loaded
                for k, v in complete_data.items():
                    sz, dtype = self.elem_infos[k]
                    assert v.shape[1:] == sz and v.dtype == dtype
                    assert verifyxor(v, check_data[k]), k
            else:
                complete_data = loaded  # BC
            self.extend(complete_data)

        self._finalize_after_load_from_folder(metadata)

    def _saver_worker(self, data_idx: int):
        assert data_idx < self.num_compelete_data, "can only save complete data"
        complete_data = self.get_complete_data(data_idx)
        check_data: Dict[str, torch.Tensor] = {}
        for k, v in complete_data.items():
            sz, dtype = self.elem_infos[k]
            assert v.shape[1:] == sz and v.dtype == dtype
            check_data[k] = checkxor(v, reducendim=len(sz))
        torch.save((complete_data, check_data), os.path.join(self.save_dir, f"{data_idx:010d}.pth"))

    def _saver_callback(self, future: concurrent.futures.Future):
        try:
            future.result()
            del self._saver_outstanding_futures[future]
        except Exception as exc:
            data_idx = self._saver_outstanding_futures[future]
            raise RuntimeError(f'Saving data at index={data_idx} errorred') from exc

    @property
    def data_num_samples(self) -> torch.Tensor:
        return self._data_num_samples_storage[:self.num_data]

    @property
    def num_compelete_samples(self) -> int:
        return self.num_samples - self.num_partial_samples

    @property
    def num_compelete_data(self) -> int:
        return self.num_data - int(self.partial_data is not None)

    @abc.abstractmethod
    def get_complete_data(self, idx: int) -> Data_t:
        pass

    def get_data(self, idx: int) -> Data_t:
        if self.partial_data is not None and idx == (self.num_data - 1):
            return self.partial_data
        else:
            return self.get_complete_data(idx)

    @abc.abstractmethod
    def _extend_complete_data(self, data: Data_t) -> Data_t:
        pass

    def _check_data_consistency(self, data: Data_t):
        assert set(data.keys()) == set(self.elem_infos.keys())
        for k, v in data.items():
            sz, dtype = self.elem_infos[k]
            assert v.shape[1:] == sz and v.dtype == dtype, (k, v, sz, dtype)

    def extend(self, data: Data_t, continue_previous: bool = False, complete: bool = True) -> Data_t:
        # we are going to handle partial data here, subclass impls can just care about complete data
        self._check_data_consistency(data)

        data_num_samples = next(iter(data.values())).shape[0]
        self.num_samples += data_num_samples
        if continue_previous:
            assert self.partial_data is not None
            if data_num_samples == 0:
                last_data_seen_so_far = self.partial_data
            else:
                last_data_seen_so_far = {
                    k: torch.cat([v, data[k]]) for k, v in self.partial_data.items()
                }
            self.data_num_samples[-1] += data_num_samples
        else:
            assert self.partial_data is None
            if self._data_num_samples_storage.shape[0] == self.num_data:
                self._data_num_samples_storage.resize_(self.num_data * 2)
            self.num_data += 1
            last_data_seen_so_far = {k: v.clone() for k, v in data.items()}  # copy!
            self.data_num_samples[-1] = data_num_samples

        if complete:
            self.partial_data = None
            self.num_partial_samples = 0

            # technically not needed (since we check in cache), but upon new complete data definitely all cache items
            # expire. so why not?
            self._sampler_cache.clear_cache()

            # from here subclasses should implement...
            complete_data = self._extend_complete_data(last_data_seen_so_far)

            data_idx = self.num_data - 1
            future = self._saver.submit(self._saver_worker, data_idx)
            self._saver_outstanding_futures[future] = data_idx
            future.add_done_callback(self._saver_callback)

            return complete_data
        else:
            self.partial_data = last_data_seen_so_far
            self.num_partial_samples += data_num_samples
            return self.partial_data

    # Never sample across episode boundary
    def batched_sample(self, batch_size: int, length: int, *,
                       include_partial_data: bool, batch_dim_first: bool = True) -> Data_t:
        num_data_in_sampling = self.num_data if include_partial_data else self.num_compelete_data
        data_max = self._sampler_cache.get(
            (include_partial_data, 'data_max', length),
            lambda: (self.data_num_samples - length).clamp_(0)[:num_data_in_sampling].numpy(),
            depends_on_partial_data=include_partial_data)

        data_p: np.ndarray = self._sampler_cache.get(
            (include_partial_data, 'p_data', length),
            lambda: data_max / data_max.sum(),
            depends_on_partial_data=include_partial_data)

        data_indices = self.np_rng.choice(
            self.num_data if include_partial_data else self.num_compelete_data,
            size=(batch_size,), replace=True, p=data_p)

        sample_indices = self.np_rng.integers(data_max[data_indices])

        sample_segs = {k: [] for k in self.elem_infos}
        for di, si in zip(data_indices, sample_indices):
            data = self.get_data(di)
            for k in self.elem_infos:
                t = data[k].narrow(0, si, length)
                assert t.shape[0] == length
                sample_segs[k].append(t)

        new_dim = 0 if batch_dim_first else 1
        return {k: torch.stack(v, dim=new_dim) for k, v in sample_segs.items()}


class ListAccessor(BaseAccessor):
    def __init__(self, elem_infos: Dict[str, Tuple[torch.Size, torch.dtype]], seed: int, save_dir: str) -> None:
        super().__init__(elem_infos, seed, save_dir)
        self.complete_data: List[BaseAccessor.Data_t] = []

    def _finalize_after_load_from_folder(self, metadata):
        super()._finalize_after_load_from_folder(metadata)
        assert len(self.complete_data) == self.num_compelete_data == self.num_data

    def get_complete_data(self, idx: int) -> BaseAccessor.Data_t:
        return self.complete_data[idx]

    def _extend_complete_data(self, data: BaseAccessor.Data_t) -> BaseAccessor.Data_t:
        self.complete_data.append(data)
        return data


class ExperienceReplay:
    r'''
    Stores sequences of (action, reward, next_observation_nonfirststep, next_observation).

    A reset is also stored in such a tuple, where `reward` is 0, and `action` is
    `next_observation_nonfirststep` (specified when creating this replay buffer).
    '''

    @dataclasses.dataclass(frozen=True)
    class Data(object):
        action: torch.Tensor
        reward: torch.Tensor
        next_observation_nonfirststep: Optional[torch.Tensor]  # whether `next_observation` is the beginning of an episode
                                                               # None means False
        next_observation: torch.Tensor

        def to(self, *args, **kwargs):
            def _optional_to(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                if x is None:
                    return x
                else:
                    return x.to(*args, **kwargs)

            return self.__class__(
                **{k: _optional_to(v) for k, v in dataclasses.asdict(self).items()}
            )

        @property
        def batch_size(self):
            return self.next_observation.shape[1]  # shape[0] is #timesteps

        @property
        def num_timesteps(self):
            return self.next_observation.shape[0]

    accessor: BaseAccessor
    _Data_cls: Type[Data]

    def __init__(self, env: AutoResetEnvBase, reset_step_action_fill_val: torch.Tensor, *,
                 sample_from_incomplete_episodes: bool = False,
                 temp_save_dir: Optional[str], seed: int):
        if temp_save_dir is None:
            temp_save_dir_obj = tempfile.TemporaryDirectory()
            atexit.register(lambda: temp_save_dir_obj.cleanup())
            temp_save_dir = temp_save_dir_obj.name

        logging.info(f'Replay buffer: temporary save dir at {temp_save_dir}')

        self.reset_step_action_fill_val = reset_step_action_fill_val
        self.env = env
        self.sample_from_incomplete_episodes = sample_from_incomplete_episodes

        assert env.observation_output_kind is not AutoResetEnvBase.ObsOutputKind.image_float32
        elem_info = dict(
            action=(torch.Size([env.action_ndim]), torch.float32),
            reward=(torch.Size([]), torch.float32),
            next_observation_nonfirststep=(torch.Size([]), torch.bool),
            next_observation=(torch.Size(env.observation_shape), env.observation_output_kind.value.dtype),
        )

        self.accessor = ListAccessor(elem_info, seed, temp_save_dir)
        self._Data_cls = ExperienceReplay.Data

    def load_from_folder(self, load_dir: str):
        self.accessor.load_from_folder(load_dir)

    @property
    def num_episodes(self):
        return self.accessor.num_data

    @property
    def num_complete_episodes(self):
        return self.accessor.num_compelete_data

    @property
    def num_samples(self):
        return self.accessor.num_samples

    @property
    def num_complete_samples(self):
        return self.accessor.num_compelete_samples

    @property
    def num_steps(self):
        # reset is not a step
        return (self.accessor.num_samples - self.accessor.num_data) * self.env.action_repeat

    @property
    def num_compelete_steps(self):
        # reset is not a step
        return (self.accessor.num_compelete_samples - self.accessor.num_compelete_data) * self.env.action_repeat

    @property
    def has_incomplete_episode(self) -> bool:
        return self.accessor.partial_data is not None

    def save_assert_all_complete(self, *, move_to_dir: Optional[str] = None):
        assert not self.has_incomplete_episode
        self.accessor.save_all_complete_data(move_to_dir=move_to_dir)

    def append_reset(self, observation: torch.Tensor):
        update_dict = dict(
            action=self.reset_step_action_fill_val.unsqueeze(0),
            reward=torch.zeros(1, dtype=torch.float32),
            next_observation_nonfirststep=torch.zeros(1, dtype=torch.bool),
            next_observation=observation.unsqueeze(0),
        )
        self.accessor.extend(
            update_dict,
            continue_previous=False,
            complete=False,
        )

    def append_step(self, action: torch.Tensor, reward: float,
                    next_observation: torch.Tensor, done: bool):
        update_dict = dict(
            action=action.unsqueeze(0),
            reward=torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
            next_observation_nonfirststep=torch.ones(1, dtype=torch.bool),
            next_observation=next_observation.unsqueeze(0),
        )
        self.accessor.extend(
            update_dict,
            continue_previous=True,
            complete=done,
        )

    def mark_previous_episode_as_complete_if_needed(self):
        if self.accessor.partial_data is not None:
            self.accessor.extend(
                {k: torch.empty(0, *s, dtype=d) for k, (s, d) in self.accessor.elem_infos.items()},
                continue_previous=True,
                complete=True,
            )
            assert self.accessor.partial_data is None

    # Returns a batch (size n) of sequence (length L) segments uniformly sampled from the memory.
    def sample(self, n, L) -> Data:
        raw_data = self.accessor.batched_sample(
            n, L, include_partial_data=self.sample_from_incomplete_episodes,
            batch_dim_first=False)
        # NOTE [ Sampled `next_observation_nonfirststep` ]
        #
        # Each segment is within an episode, so no need to reset.
        #
        # Even if we do sample the beginning of an episode, it will be the
        # beginning of a segment too, which means that it will be handled
        # identically for the  transition model (i.e., transitioning from a fake
        # all zeroes sobservation via `reset_step_action_fill_val` action). So
        # we can safely set this to be None.
        #
        # However, if replay buffer sampling is done across episode boundary,
        # this need not be always None.
        return ExperienceReplay.Data(
            action=raw_data['action'],
            reward=raw_data['reward'],
            next_observation_nonfirststep=None,
            next_observation=self.env.process_observation_as_network_input(raw_data['next_observation']),
        )



@attrs.define(kw_only=True, auto_attribs=True)
class ExperienceReplayConfig:
    _target_: str = attrs.Factory(lambda: f"{ExperienceReplay.__module__}.{ExperienceReplay.__qualname__}")
    _partial_: bool = True

    class InstantiatedT(Protocol):
        def __call__(self, env: AutoResetEnvBase, reset_step_action_fill_val: torch.Tensor, *,
                     seed: int) -> ExperienceReplay: ...

    sample_from_incomplete_episodes: bool = False  # False mimics official dreamer implementation
    temp_save_dir: Optional[str] = attrs.field(default=None)
    @temp_save_dir.validator
    def check(self, attribute, value):
        if value is not None and not os.path.exists(value):
            raise ValueError("specified temp_save_dir is not an existing path")
