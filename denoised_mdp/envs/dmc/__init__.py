# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from ..abc import EnvBase

from .dmc2gym.wrappers import DMCWrapper


VARIANTS = [
    'noiseless',
    'video_background',
    'video_background_noisy_sensor',
    'video_background_camera_jitter',
]


def make_env(spec: str, observation_output_kind: EnvBase.ObsOutputKind, seed,
             max_episode_length, action_repeat, batch_shape):
    # avoid circular imports
    from ..utils import make_batched_auto_reset_env, as_SeedSequence, get_kinetics_dir

    for variant in VARIANTS:
        if spec.endswith('_' + variant):
            break
    else:
        # if not break
        raise ValueError(f"Unexpected environment: {spec}")

    domain_name, task_name = spec[:-(len(variant) + 1)].split('_', maxsplit=1)

    kwargs = dict(
        domain_name=domain_name,
        task_name=task_name,
        observation_output_kind=observation_output_kind,
        frame_skip=action_repeat,
        total_frames=1000,
        height=64,
        width=64,
        episode_length=max_episode_length,
        # distractors
        resource_files=None,
        sensor_noise_mult=0,
        spatial_jitter=0,
        # others
        camera_id=0,
        environment_kwargs=None,
        task_kwargs={},
        visualize_reward=False,
    )

    # Sec. B.1.1
    if (domain_name, task_name) == ('walker', 'walk'):
        background_remove_mode = 'argmax'
    else:
        background_remove_mode = 'dbc'

    if variant == 'noiseless':
        pass
    elif variant == 'video_background':
        kwargs.update(
            resource_files=os.path.join(get_kinetics_dir(), 'train/driving_car/*.mp4'),
            background_remove_mode=background_remove_mode,
        )
    elif variant == 'video_background_noisy_sensor':
        kwargs.update(
            resource_files=os.path.join(get_kinetics_dir(), 'train/driving_car/*.mp4'),
            sensor_noise_mult=1,
            background_remove_mode=background_remove_mode,
        )
        if f"{domain_name}_{task_name}" not in ['cheetah_run', 'walker_walk', 'reacher_easy']:
            raise RuntimeError(f'Noisy sensor not implemented for {domain_name}_{task_name}')
    elif variant == 'video_background_camera_jitter':
        kwargs.update(
            resource_files=os.path.join(get_kinetics_dir(), 'train/driving_car/*.mp4'),
            spatial_jitter=120,
            background_remove_mode=background_remove_mode,
        )
    else:
        raise ValueError(f"Unexpected environment: {spec}")

    return make_batched_auto_reset_env(
        lambda seed: DMCWrapper(seed=as_SeedSequence(seed), **kwargs),
        seed, batch_shape)


__all__ = ['make_env']
