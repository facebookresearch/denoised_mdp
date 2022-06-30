# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import os
import sys
import tempfile
import subprocess

import enum
import attrs
import hydra.types
import hydra.utils
import hydra.core.config_store
from omegaconf import OmegaConf, MISSING, DictConfig

import torch

from denoised_mdp import utils
from denoised_mdp.envs import EnvConfig
from denoised_mdp.memory import ExperienceReplayConfig
from denoised_mdp.agents import (
    DenoisedMDP, DynamicsBackpropagateActorCritic, SoftActorCritic,
    BasePolicyLearning, VariationalModelLearning
)


@attrs.define(kw_only=True, auto_attribs=True)
class DeviceConfig:
    _target_: str = 'torch.device'

    type: str = 'cuda'
    index: Optional[int] = 0


def get_dotted_name(cls: Type):
    return f"{cls.__module__}.{cls.__qualname__}"


cs = hydra.core.config_store.ConfigStore.instance()


@attrs.define(kw_only=True, auto_attribs=True)
class Config:
    defaults: List = attrs.Factory(lambda: [
        {'learning/policy_learning': 'dynamics_backprop'},
        {'override hydra/hydra_logging': 'disabled'},  # we handle ourselves
        {'override hydra/job_logging': 'disabled'},  # we handle ourselves
        '_self_',
    ])
    _target_: str = attrs.Factory(lambda: get_dotted_name(InstantiatedConfig))

    # Disable hydra working directory creation
    hydra: Dict = dict(
        output_subdir=None,
        job=dict(chdir=False),
        run=dict(dir=tempfile.TemporaryDirectory().name),  # can't disable https://github.com/facebookresearch/hydra/issues/1937
        mode=hydra.types.RunMode.RUN,  # sigh: https://github.com/facebookresearch/hydra/issues/2262
    )

    # Directories: e.g., output

    base_git_dir: str = subprocess.check_output(
        r'git rev-parse --show-toplevel'.split(),
        cwd=os.path.dirname(__file__), encoding='utf-8').strip()

    def fmtdir(self, dir):
        return f"$GIT/{os.path.relpath(dir, self.base_git_dir)}"

    overwrite_output: bool = False
    output_base_dir: str = attrs.field(default=os.path.join(base_git_dir, 'results'))

    @output_base_dir.validator
    def exists_path(self, attribute, value):
        assert os.path.exists(value)

    output_folder: Optional[str] = None

    # Seeding
    seed: int = 60912
    test_seed: int = 1841
    device: DeviceConfig = DeviceConfig()

    # Env
    env: EnvConfig = EnvConfig()

    @attrs.define(kw_only=True, auto_attribs=True)
    class LearningConfig:
        _target_: str = attrs.Factory(lambda: get_dotted_name(InstantiatedConfig.LearningConfig))

        experience_replay: ExperienceReplayConfig = ExperienceReplayConfig()

        @attrs.define(kw_only=True, auto_attribs=True)
        class OptimizationConfig:
            _target_: str = attrs.Factory(lambda: get_dotted_name(InstantiatedConfig.LearningConfig.OptimizationConfig))

            adam_eps: float = attrs.field(default=1e-7, validator=attrs.validators.gt(0))
            train_interval: int = attrs.field(default=1000, validator=attrs.validators.gt(0))  # num steps between training
            train_iterations: int = attrs.field(default=100, validator=attrs.validators.gt(0))  # num gd steps per training
            model_every: int = attrs.field(default=1, validator=attrs.validators.gt(0))  # during train_iterations steps, frequency of model updatres
            policy_every: int = attrs.field(default=1, validator=attrs.validators.gt(0))  # during train_iterations steps, frequency of policy updatres

        optimization: OptimizationConfig = attrs.Factory(OptimizationConfig)

        model: DenoisedMDP.Config = DenoisedMDP.Config()
        model_learning: VariationalModelLearning.Config = VariationalModelLearning.Config()
        # See `defaults` list above. Related: https://github.com/facebookresearch/hydra/issues/2227
        policy_learning: BasePolicyLearning.Config = MISSING

        @attrs.define(kw_only=True, auto_attribs=True)
        class ExplorationConfig:
            _target_: str = attrs.Factory(lambda: get_dotted_name(InstantiatedConfig.LearningConfig.ExplorationConfig))

            action_noise: float = attrs.field(default=0.3, validator=attrs.validators.gt(0))
            prefill_steps: int = attrs.field(default=5000, validator=attrs.validators.gt(0))
            total_steps: int = attrs.field(default=int(1e6), validator=attrs.validators.gt(0))

        exploration: ExplorationConfig = attrs.Factory(ExplorationConfig)

        batch_size: int = attrs.field(default=50, validator=attrs.validators.gt(0))
        chunk_length: int = attrs.field(default=50, validator=attrs.validators.gt(0))

        @attrs.define(kw_only=True, auto_attribs=True)
        class TestConfig:
            _target_: str = attrs.Factory(lambda: get_dotted_name(InstantiatedConfig.LearningConfig.TestConfig))

            num_episodes: int = attrs.field(default=10, validator=attrs.validators.gt(0))
            visualize_num_episodes: int = attrs.field(default=3, validator=attrs.validators.gt(0))
            test_interval: int = attrs.field(default=10000, validator=attrs.validators.gt(0))
            visualize_interval: int = attrs.field(default=20000, validator=attrs.validators.gt(0))

        test: TestConfig = attrs.Factory(TestConfig)

        checkpoint_interval: int = attrs.field(default=250000, validator=attrs.validators.gt(0))

    learning: LearningConfig = attrs.Factory(LearningConfig)



global_config: Optional[Config] = None


@attrs.define(kw_only=True, auto_attribs=True)
class InstantiatedConfig(Config):
    config: Config = attrs.field(init=False)

    output_dir: str = attrs.field(init=False)

    # NOTE [ Pre-emption ]
    #
    # SIGUSR1 is our mechanism to pre-empt. Unpon SIGUSR1, we catch the signal,
    # start checkpointing (incl. replay buffer) via `ModelTrainer.save_resumable_state_if_possible`,
    # and finally kill the training job.
    #
    # We make 3 files/dirs for handling preemption.
    # 1. a COMPLETE file which signals that the jobs is done, so, upon SIGUSR1, no-op
    # 2. a temp CKPT dir, which we will write ckpt to upon SIGUSR1
    # 3. an actual CKPT dir, which we will copy/rename temp CKPT to, after finish writing to
    #    the temp CKPT, in case the job gets killed when we write to temp CKPT dir, leaving
    #    the ckpt in an inconsistent state.
    #
    # When we start running with the same `output_dir`, we will attempt to resume via `ModelTrainer.resume_from_saved_state_if_exists`.

    job_complete_file: str = attrs.field(init=False)
    tmp_preempt_ckpt_dir: str = attrs.field(init=False)
    preempt_ckpt_dir: str = attrs.field(init=False)
    received_SIGUSR1: bool = attrs.field(init=False)

    # logging
    logging_file: bool = attrs.field(init=False)


    def __attrs_post_init__(self):
        # Validate several entries
        assert global_config is not None
        self.config = global_config
        # Set output directory
        if self.output_folder is None:
            specs = [
                f'alpha={self.config.learning.model_learning.kl.alpha:g}',
                f'betaX={self.config.learning.model_learning.kl.beta_x:g}',
                f'betaY={self.config.learning.model_learning.kl.beta_y:g}',
                f'betaZ={self.config.learning.model_learning.kl.beta_z:g}',
                self.config.learning.policy_learning._kind_,
                f'seed={self.seed}',
            ]
            if self.config.learning.model_learning.kl.beta_free_nats_x is not None:
                specs.insert(2, f'betaFreeNatsX={self.config.learning.model_learning.kl.beta_free_nats_x:g}')
            self.output_folder = os.path.join(
                f'{self.config.env.kind.name}_{self.config.env.spec}',
                '_'.join(specs),
            )
        self.output_dir = os.path.join(self.output_base_dir, self.output_folder)
        os.makedirs(self.output_dir, exist_ok=True)
        # See NOTE [ Pre-emption ]
        self.job_complete_file = os.path.join(self.output_dir, 'COMPLETE')
        if os.path.exists(self.job_complete_file) and not self.overwrite_output:
            print(f"Complete run found at {self.output_dir} and overwrite_output=False, exiting.")
            sys.exit(0)
        self.tmp_preempt_ckpt_dir = os.path.join(self.output_dir, 'preempt.tmp')
        if os.path.exists(self.tmp_preempt_ckpt_dir):
            utils.rm_if_exists(self.tmp_preempt_ckpt_dir, maybe_dir=True)
        self.preempt_ckpt_dir = os.path.join(self.output_dir, 'preempt')
        self.received_SIGUSR1 = False
        # Logging
        self.logging_file = os.path.join(self.output_dir, 'output.log')
        utils.logging.configure(self.logging_file)

    device: torch.device
    env: EnvConfig.InstantiatedT

    @attrs.define(kw_only=True, auto_attribs=True)
    class LearningConfig(Config.LearningConfig):
        experience_replay: ExperienceReplayConfig.InstantiatedT
        model: DenoisedMDP.Config.InstantiatedT
        model_learning: VariationalModelLearning.Config.InstantiatedT
        policy_learning: BasePolicyLearning.Config.InstantiatedT

    learning: LearningConfig


cs.store(name='config', node=Config())
cs.store(group='learning/policy_learning', name='dynamics_backprop', node=DynamicsBackpropagateActorCritic.Config())
cs.store(group='learning/policy_learning', name='sac', node=SoftActorCritic.Config())


def to_config_and_instantiate(dict_cfg: DictConfig) -> Tuple[Config, InstantiatedConfig]:
    global global_config

    def convert(v: Any, desired_ty: Type):
        if isinstance(v, DictConfig):
            ty = OmegaConf.get_type(v)
            assert issubclass(ty, desired_ty)
            if attrs.has(ty):
                fields = attrs.fields_dict(ty)
                kwargs = {}
                for k, subv in v.items():
                    # sign, attrs auto-strip leading underscore so we have to manually do this
                    # rather than using `OmegaConf.to_object`
                    kwargs[k.lstrip('_')] = convert(subv, fields[k].type)
                v = ty(**kwargs)
        elif isinstance(desired_ty, type) and issubclass(desired_ty, enum.Enum):
            if isinstance(v, str) and v != MISSING:
                v = desired_ty[v]
        return v

    global_config = convert(dict_cfg, Config)
    return global_config, hydra.utils.instantiate(global_config)
