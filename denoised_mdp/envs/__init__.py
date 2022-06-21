from typing import *

import sys
if sys.version_info < (3, 8):
    from typing_extensions import Protocol

import attrs
import enum
from omegaconf import MISSING

from .abc import AutoResetEnvBase
from .utils import split_seed
from .interaction import env_interact_random_actor, env_interact_with_model, EnvInteractData


class EnvKind(enum.Enum):
    dmc = enum.auto()
    robodesk = enum.auto()
    old_robodesk = enum.auto()

    @staticmethod
    def create(kind, spec: str, action_repeat: int, max_episode_length: int, *, for_storage: bool,
               seed: int, batch_shape=()) -> AutoResetEnvBase:
        if kind is EnvKind.dmc:
            from .dmc import make_env
        elif kind is EnvKind.robodesk:
            from .robodesk import make_env
        else:
            # kind is EnvKind.old_robodesk
            def make_env(spec: str, observation_output_kind: AutoResetEnvBase.ObsOutputKind,
                         action_repeat: int, max_episode_length: int, *, seed: int, batch_shape=()):
                from .robodesk_old import parse_RoboDeskEnv
                task, noise_spec = spec.rsplit('_', 1)
                assert noise_spec == 'noisy'
                env = f"robodesk_{task.replace('_', '-')}_variant=three-neartv-bright_button=NoistyCorrelated_gL=green-light,tv-green_objL=RangeExtOffset_envL=Noisyy_cam=0.3>0.65_res=96_noiseV=0.5_sliderAlph=1_views=lessfar-topview_tv=Video2SharpContrast-FS1-Roll1"
                return parse_RoboDeskEnv(env, observation_output_kind, seed, max_episode_length, action_repeat, batch_shape)

        observation_output_kind: AutoResetEnvBase.ObsOutputKind
        if for_storage:
            observation_output_kind = AutoResetEnvBase.ObsOutputKind.image_uint8
        else:
            observation_output_kind = AutoResetEnvBase.ObsOutputKind.image_float32
        return make_env(spec, observation_output_kind=observation_output_kind, action_repeat=action_repeat,
                        max_episode_length=max_episode_length, seed=seed, batch_shape=batch_shape)



@attrs.define(kw_only=True, auto_attribs=True)
class EnvConfig:
    _target_: str = attrs.Factory(lambda: f"{EnvKind.create.__module__}.{EnvKind.create.__qualname__}")
    _partial_: bool = True

    class InstantiatedT(Protocol):
        def __call__(self, *, for_storage: bool, seed: int, batch_shape=()) -> AutoResetEnvBase: ...

    kind: EnvKind = MISSING
    spec: str = MISSING
    action_repeat: int = attrs.field(default=2, validator=attrs.validators.gt(0))
    max_episode_length: int = attrs.field(default=1000, validator=attrs.validators.gt(0))



__all__ = ['AutoResetEnvBase', 'split_seed', 'env_interact_random_actor', 'env_interact_with_model',
           'EnvInteractData']
