from typing_extensions import Protocol
from typing import *

import os
import socket
import glob
import functools
import contextlib

import torch
import numpy as np
import cv2
import gym
from PIL import Image

from .robodesk import RoboDesk as OriginalRoboDeskEnv
from ..dmc.dmc2gym import natural_imgsource

from ..abc import EnvBase, IndexableSized, AutoResetEnvBase
from ... import utils


@torch.jit.script
def _rgb2hsv(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img.unbind(dim=-3)

    # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
    # src/libImaging/Convert.c#L330
    maxc = torch.max(img, dim=-3).values
    minc = torch.min(img, dim=-3).values

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occuring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    cr = maxc - minc
    # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = hr + hg + hb
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc), dim=-3)


@torch.jit.script
def _hsv2rgb(img: torch.Tensor) -> torch.Tensor:
    h, s, v = img.unbind(dim=-3)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)

    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)

    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)

    return torch.einsum("...ijk, ...xijk -> ...xjk", mask.to(dtype=img.dtype), a4).clamp_(0, 1)


def adjust_hue(img: torch.Tensor, hue_factor: torch.Tensor) -> torch.Tensor:
    assert img.dtype == torch.uint8
    img = img.to(dtype=torch.float32) / 255.0

    if hue_factor.shape[-1] == 1 and hue_factor.ndim == 3:
        hue_factor = hue_factor[..., 0]

    img = _rgb2hsv(img)
    h, s, v = img.unbind(dim=-3)
    h = (h + hue_factor) % 1.0
    img = torch.stack((h, s, v), dim=-3)
    img_hue_adj = _hsv2rgb(img)

    return (img_hue_adj * 255.0).to(dtype=torch.uint8)


# Get camera loc from a dm_control.mujoco.engine.Pose
def pose2from(pose):
    x, y, z = pose.lookat
    flat_distance = pose.distance * np.cos(pose.elevation * np.pi / 180)
    lookfrom = np.array([
        x - flat_distance * np.cos(pose.azimuth * np.pi / 180),
        y - flat_distance * np.sin(pose.azimuth * np.pi / 180),
        z - pose.distance * np.sin(pose.elevation * np.pi / 180),
    ])
    return lookfrom


# Construct a dm_control.mujoco.engine.Pose from camera loc and target loc
def from2pose(lookfrom, lookat):
    import dm_control

    at2from = lookat - lookfrom
    distance = np.linalg.norm(at2from)
    elevation = np.arcsin(at2from[-1] / distance) * 180 / np.pi
    azimuth = np.arctan2(at2from[1], at2from[0]) * 180 / np.pi
    return dm_control.mujoco.engine.Pose(
        lookat=lookat,
        distance=distance,
        azimuth=azimuth,
        elevation=elevation,
    )


class ViewOffset(NamedTuple):
    azimuth: float = 0
    elevation: float = 0
    distance: float = 0
    lookat_x: float = 0
    lookat_y: float = 0
    lookat_z: float = 0
    cropbox_120: np.ndarray = np.array((16.75, 25.0, 105.0, 88.75))

    def get_cropbox(self, raw_render_size):
        return self.cropbox_120 * raw_render_size / 120  # h/w ~= 0.72

    def update_camera(self, cam):
        cam._render_camera.azimuth += self.azimuth
        cam._render_camera.elevation += self.elevation
        cam._render_camera.distance += self.distance
        cam._render_camera.lookat[0] += self.lookat_x
        cam._render_camera.lookat[1] += self.lookat_y
        cam._render_camera.lookat[2] += self.lookat_z
        return cam


class RoboDeskEnv(OriginalRoboDeskEnv, EnvBase):
    VIEW_OFFSETS = dict(
        far=ViewOffset(
            elevation=27,
            distance=3.3,
            lookat_x=1.275,
            lookat_y=1.35,
            lookat_z=0.525,
            cropbox_120=np.array((0, 25.892, 120, 120)),
        ),
        far_topview=ViewOffset(
            elevation=13,
            distance=2.85,
            lookat_x=1.275,
            lookat_y=1.4,
            lookat_z=0.525,
            cropbox_120=np.array((0, 25.892, 120, 120)),
        ),
        lesfar_topview=ViewOffset(
            elevation=13,
            distance=2.4,
            lookat_x=1.2,
            lookat_y=0.9,
            lookat_z=0.875,
            cropbox_120=np.array((0, 25.892, 120, 120)),
        ),
        lescfar_topview=ViewOffset(
            elevation=13,
            distance=1.8,
            lookat_x=1,
            lookat_y=0.65,
            lookat_z=0.875,
            cropbox_120=np.array((0, 25.892, 120, 120)),
        ),
        lessfar_topview=ViewOffset(
            elevation=13,
            distance=1.15,
            lookat_x=0.875,
            lookat_y=0.4,
            lookat_z=0.875,
            cropbox_120=np.array((0, 25.892, 120, 120)),
        ),
        lesscfar_topview=ViewOffset(
            elevation=13,
            distance=0.35,
            lookat_x=0.65,
            lookat_y=0.1,
            lookat_z=0.875,
            cropbox_120=np.array((0, 25.892, 120, 120)),
        ),
        lesssfar_topview=ViewOffset(
            elevation=13,
            distance=-.35,
            lookat_x=0.425,
            lookat_y=-0.25,
            lookat_z=0.875,
            cropbox_120=np.array((0, 25.892, 120, 120)),
        ),
        lessscfar_topview=ViewOffset(
            elevation=13,
            distance=-1.6,
            lookat_x=0.1,
            lookat_y=-0.65,
            lookat_z=0.875,
            cropbox_120=np.array((0, 25.892, 120, 120)),
        ),
        high=ViewOffset(),
        mid=ViewOffset(elevation=30, distance=-0.2, lookat_z=-0.175),
        low=ViewOffset(elevation=60, distance=-0.3, lookat_z=-0.2),
        low_right=ViewOffset(
            azimuth=20,
            elevation=60,
            distance=-1.2,
            lookat_x=0.325,
            lookat_y=0.25,
            lookat_z=-0.2,
        ),
    )

    TRACKABLES = {'end_effector'}

    whitenoise: Optional[torch.Tensor] = None
    reward_true_latent: Optional[torch.Tensor] = None
    button_noise: Optional[torch.Tensor] = None
    light_ids: np.ndarray
    light_mechanism: str

    def __init__(self, task='open_slide', reward='dense', action_repeat=1, episode_length=500, seed=None,
                 image_size=64, xml_file='desk.xml', camera_raw_render_size=None, resize_interpolation=Image.LANCZOS,
                 observation_output_kind=EnvBase.ObsOutputKind.image_uint8,
                 noise_speed_factor=1,  # speed steps noise per env step
                 observation_hue_change_magnitude=0,
                 observation_hue_change_smoothness=0.15,
                 light_mechanism='new',
                 button_noise_scale=0,
                 button_to_light_offset=0,
                 button_noise_iid_scale=0,
                 button_noise_smoothness=0.15,
                 red_button_functions=['red_light'],
                 green_button_functions=['green_light'],
                 blue_button_functions=['blue_light'],
                 env_light_swing_angle_mult=[1, 1, 1],
                 env_light_swing_angle_speed=[np.sqrt(5) / 15, 1 / 15, 1 / (15 * np.sqrt(3))],
                 env_light_swing_angle_stddev=[np.sqrt(5) / 60, 1 / 60, 1 / (60 * np.sqrt(3))],
                 env_light_swing_radius=[0.5, 1 / np.sqrt(2), 1],
                 env_light_swing_action_angle_speed=[0.05, 0.05, 0.05],
                 env_light_flicker_period=[1 / 30, np.sqrt(8) / 30, np.sqrt(28) / 30],
                 env_light_flicker_magnitude=[0.15, 0.15, 0.15],
                 env_light_flicker_stddev=[0.1, 0.1, 0.1],
                 env_light_flicker_base_brightness=0.3,
                 reward_diffusion_noise_scale=0,
                 reward_diffusion_noise_smoothness=0.15,
                 reward_diffusion_noise_visible_as_lights=False,
                 cam_lookfrom_jitter=None,
                 cam_lookat_jitter=None,
                 slider_panel_alpha=1,
                 views: List[str] = ['high'],
                 track_info: List[str] = [],
                 tv_image_source_fn: Optional[Callable[[], natural_imgsource.ImageSource]] = None):

        # super().__init__ calls seed(), which needs below
        self.tv_image_source = tv_image_source_fn() if tv_image_source_fn is not None else None
        super().__init__(task=task, reward=reward, action_repeat=action_repeat, episode_length=episode_length,
                         seed=seed, image_size=image_size, xml_file=xml_file,
                         camera_raw_render_size=camera_raw_render_size, resize_interpolation=resize_interpolation)
        self._observation_output_kind = observation_output_kind
        self.observation_hue_change_magnitude = observation_hue_change_magnitude
        self.observation_hue_change_smoothness = observation_hue_change_smoothness

        if noise_speed_factor != 1:
            self.observation_hue_change_smoothness = 1 - (1 - self.observation_hue_change_smoothness) ** noise_speed_factor

        assert light_mechanism in {'original', 'new', 'new+range', 'new+range+ext', 'new+range+ext+negonly'}
        self.c2idx = {c: i for i, c in enumerate(['red', 'green', 'blue'])}
        self.light_mechanism = light_mechanism
        if light_mechanism != 'original':
            for c in ['red', 'green', 'blue']:
                self.physics.named.model.geom_rgba[f'{c}_light_rise_cylinder'][-1] = 0
                # self.physics.named.model.geom_rgba[f'{c}_light_overlay'][-1] = 0
        self.reward_diffusion_noise_scale = reward_diffusion_noise_scale
        self.reward_diffusion_noise_smoothness = reward_diffusion_noise_smoothness
        self.reward_diffusion_noise_visible_as_lights = reward_diffusion_noise_visible_as_lights

        if noise_speed_factor != 1:
            self.reward_diffusion_noise_smoothness = 1 - (1 - self.reward_diffusion_noise_smoothness) ** noise_speed_factor

        self.button_noise_scale = button_noise_scale
        self.button_to_light_offset = torch.as_tensor(button_to_light_offset, dtype=torch.float32).expand(3)
        self.button_noise_iid_scale = button_noise_iid_scale
        self.button_noise_smoothness = button_noise_smoothness
        self.light_ids = []
        for c in ['red', 'green', 'blue']:
            cids = []
            for suff in ['collision', 'rise_cylinder', 'overlay', 'neg_overlay', 'background']:
                cids.append(self.physics.model.name2id(f'{c}_light_{suff}', 'geom'))
            self.light_ids.append(cids)
        self.light_ids = np.asarray(self.light_ids)

        if noise_speed_factor != 1:
            self.button_noise_smoothness = 1 - (1 - self.button_noise_smoothness) ** noise_speed_factor
            # self.button_noise_iid_scale.mul_(noise_speed_factor)  # not accumulative

        self.env_light_swing_angle_mult = torch.as_tensor(env_light_swing_angle_mult, dtype=torch.float32).expand(3)
        self.env_light_swing_angle_speed = torch.as_tensor(env_light_swing_angle_speed, dtype=torch.float32).expand(3)
        assert (self.env_light_swing_angle_speed >= self.ENV_LIGHT_SWING_ANGLE_SPEED_MIN).all()
        assert (self.env_light_swing_angle_speed <= self.ENV_LIGHT_SWING_ANGLE_SPEED_MAX).all()
        self.env_light_swing_angle_stddev = torch.as_tensor(env_light_swing_angle_stddev, dtype=torch.float32).expand(3)
        self.env_light_swing_radius = torch.as_tensor(env_light_swing_radius, dtype=torch.float32).expand(3)
        if env_light_swing_action_angle_speed is None:
            self.env_light_swing_action_angle_speed = None
        else:
            self.env_light_swing_action_angle_speed = torch.as_tensor(env_light_swing_action_angle_speed, dtype=torch.float32).expand(3)
            self.action_dim += 3
        self.env_light_flicker_period = torch.as_tensor(env_light_flicker_period, dtype=torch.float32).expand(3)
        self.env_light_flicker_magnitude = torch.as_tensor(env_light_flicker_magnitude, dtype=torch.float32).expand(3)
        self.env_light_flicker_stddev = torch.as_tensor(env_light_flicker_stddev, dtype=torch.float32).expand(3)
        self.env_light_flicker_base_brightness = torch.as_tensor(env_light_flicker_base_brightness, dtype=torch.float32).expand(3)

        if noise_speed_factor != 1:
            self.env_light_swing_angle_speed.mul_(noise_speed_factor)
            self.env_light_swing_angle_stddev.mul_(noise_speed_factor)  # accumulative
            self.env_light_flicker_period.div_(noise_speed_factor)
            # self.env_light_flicker_stddev.mul_(noise_speed_factor)  # not accumulative

        assert set(views).issubset(self.VIEW_OFFSETS.keys())
        self.views = views
        assert len(views) > 0

        assert set(track_info).issubset(self.TRACKABLES)
        self.track_info = track_info

        if cam_lookfrom_jitter is not None:
            self.cam_lookfrom_jitter = torch.as_tensor(cam_lookfrom_jitter, dtype=torch.float32).expand(3)
        else:
            self.cam_lookfrom_jitter = None

        if cam_lookat_jitter is not None:
            self.cam_lookat_jitter = torch.as_tensor(cam_lookat_jitter, dtype=torch.float32).expand(3)
        else:
            self.cam_lookat_jitter = None

        self.tv_texid = self.physics.named.model.mat_texid['tv_material']
        # if self.tv_image_source is None:
        #     self.physics.named.model.geom_rgba['tv_bezel'][-1] = 0
        #     self.physics.named.model.geom_rgba['tv_screen'][-1] = 0

        self.physics.named.model.geom_rgba['desk_slide_panel'][-1] = slider_panel_alpha
        self.noise_speed_factor = noise_speed_factor

        self.red_button_functions: Set[str] = set(red_button_functions)
        assert self.red_button_functions.issubset({'red_light', 'tv_red', 'tv_contrast', 'tv_brightness', 'tv_speed'})
        self.green_button_functions: Set[str] = set(green_button_functions)
        assert self.green_button_functions.issubset({'green_light', 'tv_green', 'tv_contrast', 'tv_brightness', 'tv_speed'})
        self.blue_button_functions: Set[str] = set(blue_button_functions)
        assert self.blue_button_functions.issubset({'blue_light', 'tv_blue', 'tv_contrast', 'tv_brightness', 'tv_speed'})
        self.buttons_affect_lights = torch.as_tensor(
            [
                'red_light' in self.red_button_functions,
                'green_light' in self.green_button_functions,
                'blue_light' in self.blue_button_functions,
            ],
            dtype=torch.bool,
        )
        if self.tv_image_source is not None:
            self.tv_adjust_rgb_mult = torch.as_tensor(
                [
                    'tv_red' in self.red_button_functions,
                    'tv_green' in self.green_button_functions,
                    'tv_blue' in self.blue_button_functions,
                ],
                dtype=torch.bool,
            )
            if not self.tv_adjust_rgb_mult.any():
                self.tv_adjust_rgb_mult = None
            else:
                assert self.tv_image_source.shape[-1] == 3
        else:
            self.tv_adjust_rgb_mult = None

        self.tv_contrast_button_source = None
        if 'tv_contrast' in self.red_button_functions:
            assert self.tv_contrast_button_source is None and self.tv_image_source is not None
            self.tv_contrast_button_source = 'red'
        if 'tv_contrast' in self.green_button_functions:
            assert self.tv_contrast_button_source is None and self.tv_image_source is not None
            self.tv_contrast_button_source = 'green'
        if 'tv_contrast' in self.blue_button_functions:
            assert self.tv_contrast_button_source is None and self.tv_image_source is not None
            self.tv_contrast_button_source = 'blue'

        self.tv_brightness_button_source = None
        if 'tv_brightness' in self.red_button_functions:
            assert self.tv_brightness_button_source is None and self.tv_image_source is not None
            self.tv_brightness_button_source = 'red'
        if 'tv_brightness' in self.green_button_functions:
            assert self.tv_brightness_button_source is None and self.tv_image_source is not None
            self.tv_brightness_button_source = 'green'
        if 'tv_brightness' in self.blue_button_functions:
            assert self.tv_brightness_button_source is None and self.tv_image_source is not None
            self.tv_brightness_button_source = 'blue'

        self.tv_speed_button_source = None
        if 'tv_speed' in self.red_button_functions:
            assert self.tv_speed_button_source is None and self.tv_image_source is not None
            self.tv_speed_button_source = 'red'
        if 'tv_speed' in self.green_button_functions:
            assert self.tv_speed_button_source is None and self.tv_image_source is not None
            self.tv_speed_button_source = 'green'
        if 'tv_speed' in self.blue_button_functions:
            assert self.tv_speed_button_source is None and self.tv_image_source is not None
            self.tv_speed_button_source = 'blue'

    @utils.lazy_property
    def reward_functions(self):
        rf = super().reward_functions
        rf.update(
            light_red=(
                lambda reward_type: self._light_reward('red', reward_type)),
            light_green=(
                lambda reward_type: self._light_reward('green', reward_type)),
            light_green_visible=(
                lambda reward_type: self._light_reward('green', reward_type, light_reward_visible=True)),
            light_blue=(
                lambda reward_type: self._light_reward('blue', reward_type)),
            light_red_nodist=(
                lambda reward_type: self._light_reward('red', reward_type, no_dist=True)),
            light_green_nodist=(
                lambda reward_type: self._light_reward('green', reward_type, no_dist=True)),
            light_blue_nodist=(
                lambda reward_type: self._light_reward('blue', reward_type, no_dist=True)),
            button_green=(
                lambda reward_type: self._new_button_reward('green', reward_type)),
            tv_green=(
                lambda reward_type: self._tv_color_reward('green', reward_type)),
            tv_brightness_70=(
                lambda reward_type: self._tv_brightness_reward(0.7, reward_type)),
            tv_brightness_100=(
                lambda reward_type: self._tv_brightness_reward(1, reward_type)),
            tv_contrast=(
                lambda reward_type: self._tv_contrast_reward(reward_type)),
            tv_contrast_w08=(
                lambda reward_type: self._tv_contrast_reward(reward_type, contrast_weight=0.8)),
        )
        return rf

    @property
    def reward_depends_on_tv_content(self) -> bool:
        return self.task in {'tv_green', 'tv_brightness_70', 'tv_brightness_100', 'tv_contrast', 'tv_contrast_w08'}

    def randn(self, *size):
        return torch.as_tensor(self.np_rng.normal(size=size), dtype=torch.float32)

    def rand(self, *size):
        return torch.as_tensor(self.np_rng.uniform(size=size), dtype=torch.float32)

    def seed(self, seed: Union[int, np.random.SeedSequence, None] = None):
        from ..utils import as_SeedSequence
        ss_other, ss_tv_image = as_SeedSequence(seed).spawn(2)
        super().seed(ss_other)
        if self.tv_image_source is not None:
            self.tv_image_source.seed(ss_tv_image)

    def get_random_state(self):
        if self.tv_image_source is not None:
            tv_image_source_rs = self.tv_image_source.get_random_state()
        else:
            tv_image_source_rs = None
        return (
            super().get_random_state(),
            tv_image_source_rs,
        )

    def set_random_state(self, random_state):
        super().set_random_state(random_state[0])
        if self.tv_image_source is not None:
            self.tv_image_source.set_random_state(random_state[1])

    ENV_LIGHT_SWING_ANGLE_SPEED_MIN = 0
    ENV_LIGHT_SWING_ANGLE_SPEED_MAX = 0.8
    env_light_swing_angle_offset: torch.Tensor
    env_light_flicker_period_offset: torch.Tensor
    env_light_diffuse0: torch.Tensor = None

    env_light_pos0: torch.Tensor = None
    env_light_pos0_sqnorm: torch.Tensor = None
    current_env_light_swing_angle: torch.Tensor = None
    current_env_light_swing_angle_speed: torch.Tensor

    def reset_env_light(self):
        self.current_env_light_swing_angle_speed = self.env_light_swing_angle_speed.clone()
        self.env_light_swing_angle_offset = self.randn(3)
        self.env_light_flicker_period_offset = self.randn(3)
        self.current_env_light_swing_angle = self.randn(3)  # self.env_light_flicker_period_offset.clone()
        if self.env_light_diffuse0 is None:
            self.env_light_diffuse0 = torch.as_tensor(self.physics.model.light_diffuse[:, 0]).clone()
        if self.env_light_pos0 is None:
            self.env_light_pos0 = torch.as_tensor(self.physics.model.light_pos).clone()
        if self.env_light_pos0_sqnorm is None:
            self.env_light_pos0_sqnorm = self.env_light_pos0.pow(2).sum(-1)

    def update_env_light(self, action):
        if self.env_light_swing_action_angle_speed is not None:
            delta: torch.Tensor = torch.as_tensor(action[-3:], dtype=torch.float32) * self.env_light_swing_action_angle_speed
            self.current_env_light_swing_angle_speed.add_(delta).clamp_(
                self.ENV_LIGHT_SWING_ANGLE_SPEED_MIN,
                self.ENV_LIGHT_SWING_ANGLE_SPEED_MAX,
            )
        self.current_env_light_swing_angle.add_(self.current_env_light_swing_angle_speed)
        self.current_env_light_swing_angle.add_(self.env_light_swing_angle_stddev * self.randn(3))
        self.current_env_light_swing_angle.fmod_(2 * np.pi)

    def emit_env_light(self):
        t = self.num_steps
        # flicker, i.e., brightness
        brs = torch.sin(
            self.env_light_flicker_period_offset + t * self.env_light_flicker_period
        ).mul_(self.env_light_flicker_magnitude).add_(self.env_light_flicker_stddev * self.randn(3))
        brs = brs.add_(self.env_light_flicker_base_brightness).clamp_(0, 1)
        # swing
        theta = self.current_env_light_swing_angle * self.env_light_swing_angle_mult
        loc_delta = torch.stack([theta.sin(), theta.cos()], dim=-1).mul(self.env_light_swing_radius[:, None])
        self.physics.model.light_pos[:, :2] = loc_delta + self.env_light_pos0[:, :2]
        light_pos_t = torch.as_tensor(self.physics.model.light_pos, dtype=torch.float32)
        # add swing effect to brightness
        distance_sq_ratio = self.env_light_pos0_sqnorm / light_pos_t.pow(2).sum(dim=-1)
        brs = brs * distance_sq_ratio

        for ii in range(3):
            self.physics.model.light_diffuse[ii, :] = brs[ii]

        light_dir = -light_pos_t / light_pos_t.norm(dim=-1, keepdim=True)
        self.physics.model.light_dir[:] = light_dir.numpy()

    def reset_button_noise(self):
        if self.button_noise_scale > 0:
            self.button_noise = self.randn(3)

    def _update_button_noise(self):
        if self.button_noise_scale > 0:
            assert 0 < self.button_noise_smoothness < 1
            torch.lerp(
                self.button_noise,
                self.randn(*self.button_noise.shape),
                weight=self.button_noise_smoothness,
                out=self.button_noise,
            )

    normalized_button: Optional[torch.Tensor] = None
    BUTTON_JNT_THRESHOLD = 0.0045
    BUTTON_JNT_MAX = 0.005
    BUTTON_JNT_THRESHOLD_NORM = BUTTON_JNT_THRESHOLD / BUTTON_JNT_MAX

    def _read_normalized_button_from_physics(self) -> torch.Tensor:
        # get button joint from negated light joints
        button_jnt = -torch.as_tensor([
            self.physics.named.data.qpos['red_light'][0],
            self.physics.named.data.qpos['green_light'][0],
            self.physics.named.data.qpos['blue_light'][0],
        ], dtype=torch.float32)
        # Convert JNT via linear tsfm
        return button_jnt / self.BUTTON_JNT_MAX

    def update_normalized_button(self):
        assert self.normalized_button is None
        self.normalized_button = self._read_normalized_button_from_physics()
        # self.normalized_button = torch.ones(3).mul(self.num_steps / 3).sin().div(2).add(0.5)
        self._update_button_noise()
        if self.button_noise_scale > 0:
            self.normalized_button.add_(self.button_noise.mul(self.button_noise_scale))
        if self.button_noise_iid_scale > 0:
            self.normalized_button.add_(self.randn(3).mul(self.button_noise_iid_scale))

        # return (neg_jnt - self.LIGHT_NEG_JNT_THRESHOLD) * self.LIGHT_NEG_JNT_RATIO + self.LIGHT_NEG_JNT_THRESHOLD_NORM_TARGET

    def before_each_physics_step(self, action):
        self.normalized_button = None  # wipe
        self.update_env_light(action)
        self.emit_env_light()

    light_alpha: Optional[torch.Tensor] = None
    LIGHT_NEG_JNT_THRESHOLD_BRIGHTNESS = 0.4  # what threshold should correspond to in terms of brightness
    LIGHT_NEG_JNT_MAX_BRIGHTNESSS = 1
    LIGHT_NEG_JNT_NORM_TO_BRIGHTNESS_RATIO = (1 - LIGHT_NEG_JNT_THRESHOLD_BRIGHTNESS) / (1 - BUTTON_JNT_THRESHOLD_NORM)
    # LIGHT_NEG_JNT_NORM_THRESHOLD = LIGHT_NEG_JNT_THRESHOLD / LIGHT_NEG_JNT_MAX

    def emit_light_mechanism(self):
        assert self.normalized_button is not None
        button_to_light = self.normalized_button + self.button_to_light_offset
        if self.light_mechanism == 'new+range':
            # BUTTON_JNT_THRESHOLD_NORM -> 0.4, 1 -> 1
            self.light_alpha = button_to_light - self.BUTTON_JNT_THRESHOLD_NORM
            self.light_alpha.mul_(self.LIGHT_NEG_JNT_NORM_TO_BRIGHTNESS_RATIO)
            self.light_alpha = self.light_alpha.add_(self.LIGHT_NEG_JNT_THRESHOLD_BRIGHTNESS).clamp_(0, 1) * self.buttons_affect_lights
            for ci, c in enumerate(['red', 'green', 'blue']):
                self.physics.named.model.geom_rgba[f'{c}_light_overlay'][-1] = self.light_alpha[ci].item()
        elif self.light_mechanism == 'new+range+ext':
            # BUTTON_JNT_THRESHOLD_NORM -> 0.4, 1 -> 1
            self.light_alpha = button_to_light - self.BUTTON_JNT_THRESHOLD_NORM
            self.light_alpha.mul_(self.LIGHT_NEG_JNT_NORM_TO_BRIGHTNESS_RATIO)
            self.light_alpha = self.light_alpha.add_(self.LIGHT_NEG_JNT_THRESHOLD_BRIGHTNESS).clamp_(-1, 1) * self.buttons_affect_lights
            for ci, c in enumerate(['red', 'green', 'blue']):
                a = self.light_alpha[ci].item()
                self.physics.named.model.geom_rgba[f'{c}_light_overlay'][-1] = max(a, 0)
                self.physics.named.model.geom_rgba[f'{c}_light_neg_overlay'][-1] = max(-a, 0)
        elif self.light_mechanism == 'new+range+ext+negonly':
            # BUTTON_JNT_THRESHOLD_NORM -> 0.4, 1 -> 1
            self.light_alpha = button_to_light - self.BUTTON_JNT_THRESHOLD_NORM
            self.light_alpha.mul_(self.LIGHT_NEG_JNT_NORM_TO_BRIGHTNESS_RATIO)
            self.light_alpha = self.light_alpha.add_(self.LIGHT_NEG_JNT_THRESHOLD_BRIGHTNESS).clamp_(-1, 0) * self.buttons_affect_lights
            for ci, c in enumerate(['red', 'green', 'blue']):
                a = self.light_alpha[ci].item()
                self.physics.named.model.geom_rgba[f'{c}_light_overlay'][-1] = 0
                self.physics.named.model.geom_rgba[f'{c}_light_neg_overlay'][-1] = max(-a, 0)
        elif self.light_mechanism == 'new':
            # (BUTTON_JNT_THRESHOLD_NORM, +\infty] -> 0.4
            light_alpha = (button_to_light > self.LIGHT_NEG_JNT_THRESHOLD_BRIGHTNESS) * 0.4
            self.light_alpha = light_alpha * self.buttons_affect_lights
            for ci, c in enumerate(['red', 'green', 'blue']):
                self.physics.named.model.geom_rgba[f'{c}_light_overlay'][-1] = self.light_alpha[ci].item()
        else:
            self.light_alpha = button_to_light

    def after_each_physics_step(self, action):
        self.update_whitenoise()
        self.update_reward_noise()
        self.update_normalized_button()  # affects light and tv!
        self.update_and_emit_camera_jitter()
        self.emit_light_mechanism()
        self.update_tv()

    def _light_reward(self, color, reward_type='dense_reward', no_dist=False, no_button=False,
                      light_reward_visible=False):
        if reward_type == 'success':
            no_dist = no_button = True

        if light_reward_visible:
            assert self.light_alpha is not None
            ctns_light_on = self.light_alpha[self.c2idx[color]].item()
        else:
            assert self.normalized_button is not None
            ctns_light_on = self.normalized_button[self.c2idx[color]].item()

        total_weights = 0.5
        total_reward = 0.5 * ctns_light_on
        if not no_dist:
            dist_reward = self._get_dist_reward(
                self.physics.named.data.xpos[color + '_button'])
            total_weights += 0.25
            total_reward += 0.25 * dist_reward
        if not no_button:
            press_button = (
                self.physics.named.data.qpos[color + '_light'][0] < -0.00453)
            total_weights += 0.25
            total_reward += 0.25 * press_button

        return total_reward / total_weights

    @utils.lazy_property
    def square_frequency(self) -> torch.Tensor:
        x = y = torch.linspace(-1, 1, self.camera_raw_render_size)
        return (x[:, None] ** 2 + y ** 2)

    def get_unit_interval_pinknoise(self, whitenoise: torch.Tensor, alpha: float = 2) -> torch.Tensor:
        ft: torch.Tensor = torch.fft.fftshift(torch.fft.fft2(whitenoise, dim=(0, 1)))
        pft = ft / self.square_frequency.pow(alpha / 2)
        pn: torch.Tensor = torch.fft.ifft2(torch.fft.ifftshift(pft), dim=(0, 1)).real
        if pn.ndim == 3:
            pn = pn - pn.flatten(0, 1).min(dim=0).values[None, None, :]
            pn = pn / pn.flatten(0, 1).max(dim=0).values[None, None, :]
        else:
            pn = pn - pn.min()
            pn = pn / pn.max()
        return pn

    def reset_whitenoise(self):
        if self.observation_hue_change_magnitude > 0:
            self.whitenoise = self.randn(self.camera_raw_render_size, self.camera_raw_render_size)

    def update_whitenoise(self):
        if self.observation_hue_change_magnitude > 0:
            assert 0 < self.observation_hue_change_smoothness < 1
            torch.lerp(
                self.whitenoise,
                self.randn(*self.whitenoise.shape),
                self.observation_hue_change_smoothness,
                out=self.whitenoise,
            )

    def reset_reward_noise(self):
        if self.reward_diffusion_noise_scale > 0 or self.reward_diffusion_noise_visible_as_lights:
            self.reward_true_latent = self.randn(3)

    def update_reward_noise(self):
        if self.reward_diffusion_noise_scale > 0 or self.reward_diffusion_noise_visible_as_lights:
            assert 0 < self.reward_diffusion_noise_smoothness < 1
            torch.lerp(
                self.reward_true_latent,
                self.randn(*self.reward_true_latent.shape),
                self.reward_diffusion_noise_smoothness,
                out=self.reward_true_latent,
            )

    def get_reward_noise_delta(self):
        if self.reward_diffusion_noise_scale > 0:
            return self.reward_true_latent.mean().item() * self.reward_diffusion_noise_scale
        return 0

    def reset_camera_jitter(self):
        from ..utils import SmoothRandomWalker
        if self.cam_lookfrom_jitter is not None:
            self.cam_lookfrom_jitter_randomwalker_gen = {
                v: iter(SmoothRandomWalker(d=3, dtmult=self.cam_lookfrom_jitter, env=self, speed_factor=self.noise_speed_factor)) for v in self.views
            }
        if self.cam_lookat_jitter is not None:
            self.cam_lookat_jitter_randomwalker_gen = {
                v: iter(SmoothRandomWalker(d=3, dtmult=self.cam_lookat_jitter, env=self, speed_factor=self.noise_speed_factor)) for v in self.views
            }
        self.update_and_emit_camera_jitter()

    def update_and_emit_camera_jitter(self):
        if self.cam_lookfrom_jitter is not None:
            self.current_cam_lookfrom_jitter = {
                v: next(g) for v, g in self.cam_lookfrom_jitter_randomwalker_gen.items()
            }
        if self.cam_lookat_jitter is not None:
            self.current_cam_lookat_jitter = {
                v: next(g) for v, g in self.cam_lookat_jitter_randomwalker_gen.items()
            }

    def _update_tv_tex(self):
        if self.tv_image_source is not None and not self.tv_tex_updated:
            contrast_delta_delta = None
            if self.tv_contrast_button_source is not None:
                # 0 -> -48, 1 -> 96
                contrast_delta_delta = self.normalized_button[self.c2idx[self.tv_contrast_button_source]]
                contrast_delta_delta = contrast_delta_delta * (96 + 48) - 48
                contrast_delta_delta = max(-127, min(127, contrast_delta_delta))

            img = self.tv_image_source.get_image(contrast_delta_delta=contrast_delta_delta)

            if self.tv_brightness_button_source is not None:
                # 0 -> 0.5, 1 -> 2.5
                # min = 0.25
                br_mul = max(0.2, self.normalized_button[self.c2idx[self.tv_brightness_button_source]] * 2 + 0.5)
                img = torch.as_tensor(img, dtype=torch.float32).mul_(br_mul).round_().clamp_(0, 255).to(torch.uint8).numpy()

            if self.tv_adjust_rgb_mult is not None:
                assert img.shape[-1] == 3
                # 0 -> 0, 1 -> 0.6
                lerp_w = (self.normalized_button * self.tv_adjust_rgb_mult).mul_(0.75).clamp_(0, 1)
                img_t: torch.Tensor = torch.lerp(
                    torch.as_tensor(img, dtype=torch.float32),
                    torch.full((), 255, dtype=torch.float32).expand(img.shape),
                    lerp_w,
                )
                img = img_t.round_().to(torch.uint8).numpy()
            if img.shape[-1] == 1:
                # for stride compatibility reasons (sigh, cv2), we have to make a new copy
                img_resized = cv2.resize(
                    img,
                    self.tv_tex.shape[:2][::-1],
                    interpolation=cv2.INTER_NEAREST,
                )
                self.tv_tex[:] = img_resized[..., None]
            else:
                cv2.resize(
                    img,
                    self.tv_tex.shape[:2][::-1],
                    dst=self.tv_tex,
                    interpolation=cv2.INTER_NEAREST,
                )
            self.tv_tex_updated = True
            self.tv_tex_emitted = False

    tv_tex_updated: bool = False
    tv_tex_emitted: bool = False

    def reset_tv(self):
        if self.tv_image_source is not None:
            self.tv_image_source.reset()
            tex_adr = self.physics.named.model.tex_adr['tv_texture']
            tex_w = self.physics.named.model.tex_width['tv_texture']
            tex_h = self.physics.named.model.tex_height['tv_texture']
            size = tex_w * tex_h * 3
            full_tex = self.physics.named.model.tex_rgb[tex_adr:tex_adr + size].reshape(tex_h, tex_w, 3)
            front_h0 = tex_h * 3 // 6
            front_ht = tex_h * 4 // 6
            full_tex[:front_h0] = 0
            full_tex[front_ht:] = 0
            self.tv_tex = full_tex[front_h0:front_ht]
            self.tv_tex_updated = self.tv_tex_emitted = False

    def update_tv(self):
        if self.tv_image_source is not None:
            increment = 1
            if self.tv_speed_button_source is not None:
                # 0 -> 0.5, 1 -> 2.5
                increment = self.normalized_button[self.c2idx[self.tv_speed_button_source]]
                increment = increment * 2 + 0.5
            self.tv_image_source.increment(increment)
            self.tv_tex_updated = self.tv_tex_emitted = False
            if self.reward_depends_on_tv_content:
                self._update_tv_tex()

    def emit_tv(self):
        if self.tv_image_source is not None:
            if not self.tv_tex_updated:
                self._update_tv_tex()
            if not self.tv_tex_emitted:
                from dm_control.mujoco.wrapper.mjbindings import mjlib
                # push updated tex to GPU
                with self.physics.contexts.gl.make_current() as ctx:
                    ctx.call(
                        mjlib.mjr_uploadTexture,
                        self.physics.model.ptr,
                        self.physics.contexts.mujoco.ptr,
                        self.tv_texid,
                    )
                self.tv_tex_emitted = True

    def _new_button_reward(self, color, reward_type='dense_reward', no_dist=False, no_button=False):
        if reward_type == 'success':
            no_dist = no_button = True

        assert self.normalized_button is not None

        total_weights = 0.5
        total_reward = 0.5 * self.normalized_button[self.c2idx[color]].item()
        if not no_dist:
            dist_reward = self._get_dist_reward(
                self.physics.named.data.xpos[color + '_button'])
            total_weights += 0.25
            total_reward += 0.25 * dist_reward
        if not no_button:
            press_button = (
                self.physics.named.data.qpos[color + '_light'][0] < -0.00453)
            total_weights += 0.25
            total_reward += 0.25 * press_button

        return total_reward / total_weights

    def _tv_color_reward(self, color, reward_type='dense_reward', no_dist=False, no_button=False):
        if reward_type == 'success':
            no_dist = no_button = True

        assert self.tv_image_source is not None

        total_weights = 0.5
        total_reward = 0.5 * self.tv_tex[..., self.c2idx[color]].mean() / 255
        if not no_dist:
            dist_reward = self._get_dist_reward(
                self.physics.named.data.xpos[color + '_button'])
            total_weights += 0.25
            total_reward += 0.25 * dist_reward
        if not no_button:
            press_button = (
                self.physics.named.data.qpos[color + '_light'][0] < -0.00453)
            total_weights += 0.25
            total_reward += 0.25 * press_button

        return total_reward / total_weights

    def _tv_brightness_reward(self, target, reward_type='dense_reward', no_dist=False, no_button=False):
        if reward_type == 'success':
            no_dist = no_button = True

        assert self.tv_image_source is not None
        assert self.tv_brightness_button_source is not None

        r = self.tv_tex[..., 0]
        g = self.tv_tex[..., 1]
        b = self.tv_tex[..., 2]
        br = (0.2989 * r + 0.587 * g + 0.114 * b) / 255
        if br > target:
            norm_closeness = 1 - (br - target) / (1 - target)
        else:
            norm_closeness = br / target

        total_weights = 0.5
        total_reward = 0.5 * (norm_closeness ** 2)
        if not no_dist:
            dist_reward = self._get_dist_reward(
                self.physics.named.data.xpos[self.tv_brightness_button_source + '_button'])
            total_weights += 0.25
            total_reward += 0.25 * dist_reward
        if not no_button:
            press_button = (
                self.physics.named.data.qpos[self.tv_brightness_button_source + '_light'][0] < -0.00453)
            total_weights += 0.25
            total_reward += 0.25 * press_button

        return total_reward / total_weights

    def _tv_contrast_reward(self, reward_type='dense_reward', no_dist=False, no_button=False,
                            contrast_weight=0.5):
        if reward_type == 'success':
            no_dist = no_button = True

        assert self.tv_image_source is not None
        assert self.tv_contrast_button_source is not None

        r = self.tv_tex[..., 0]
        g = self.tv_tex[..., 1]
        b = self.tv_tex[..., 2]
        I = (0.2989 * r + 0.587 * g + 0.114 * b) / 255
        # rms contrast in [0, 0.5]
        contrast = np.std(I)

        total_weights = contrast_weight
        total_reward = contrast_weight * contrast * 2
        if not no_dist:
            dist_reward = self._get_dist_reward(
                self.physics.named.data.xpos[self.tv_contrast_button_source + '_button'])
            w = (1 - contrast_weight) / 2
            total_weights += w
            total_reward += w * dist_reward
        if not no_button:
            press_button = (
                self.physics.named.data.qpos[self.tv_contrast_button_source + '_light'][0] < -0.00453)
            w = (1 - contrast_weight) / 2
            total_weights += w
            total_reward += w * press_button

        return total_reward / total_weights

    def reset_after_physics_before_rendering(self):
        self.reset_env_light()
        self.reset_whitenoise()
        self.reset_reward_noise()
        self.reset_button_noise()
        self.normalized_button = None
        self.update_normalized_button()
        self.emit_env_light()
        self.emit_light_mechanism()
        self.reset_camera_jitter()
        self.reset_tv()

    _pose2from = staticmethod(pose2from)
    _from2pose = staticmethod(from2pose)

    def _create_camera(self, view=None):
        if view is None:
            view = self.views[0]
        else:
            assert view in self.views
        if self.cam_lookfrom_jitter is None and self.cam_lookat_jitter is None:
            return super()._create_camera()
        camera = super()._create_camera(movable=True)
        pose = camera.get_pose()
        lookat = pose.lookat
        lookfrom = pose2from(pose)
        if self.cam_lookat_jitter is not None:
            lookat += np.asarray(self.current_cam_lookat_jitter[view])
        if self.cam_lookfrom_jitter is not None:
            lookfrom += np.asarray(self.current_cam_lookfrom_jitter[view])
        pose = from2pose(lookfrom, lookat)
        camera.set_pose(*pose)
        camera = self.VIEW_OFFSETS[view].update_camera(camera)
        return camera

    def _get_task_reward(self, task, reward_type):
        return super()._get_task_reward(task, reward_type) + self.get_reward_noise_delta()

    def reset(self) -> Tuple[torch.Tensor, EnvBase.Info]:
        return super().reset(), EnvBase.Info(0)

    def step(self, action) -> Tuple[torch.Tensor, float, bool, EnvBase.Info]:
        if isinstance(action, torch.Tensor):
            action = action.detach()
        obs, reward, done, info = super().step(np.asarray(action))
        info = EnvBase.Info(
            info['actual_env_steps_taken'],
        )
        return obs, reward, done, info

    @contextlib.contextmanager
    def _as_observation_output_kind(self, observation_output_kind=None):
        if observation_output_kind is None:
            yield
        else:
            old = self._observation_output_kind
            self._observation_output_kind = observation_output_kind
            yield
            self._observation_output_kind = old

    def _get_obs(self, *, observation_output_kind=None):
        with self._as_observation_output_kind(observation_output_kind):
            # if self.symbolic:
            #     return torch.as_tensor(self._get_symbolic_state(), dtype=torch.float32)
            # else:
            self.emit_tv()
            images: List[torch.Tensor] = []
            for view in self.views:
                camera = self._create_camera(view=view)
                image: np.ndarray = self.render(resize=False, camera=camera)
                if self.observation_hue_change_magnitude > 0 or self.reward_diffusion_noise_scale > 0:
                    assert False

                    image_t: torch.Tensor = torch.as_tensor(image.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0

                    image_t = _rgb2hsv(image_t)
                    h, s, v = image_t.unbind(dim=-3)

                    if self.observation_hue_change_magnitude > 0:
                        pn = self.get_unit_interval_pinknoise(self.whitenoise)
                        hue_factor = pn.sub_(0.5).mul_(self.observation_hue_change_magnitude)
                        h = (h + hue_factor) % 1.0

                    if self.reward_diffusion_noise_visible_as_lights:
                        seg_im = camera.render(depth=False, segmentation=True)
                        seg_im = torch.as_tensor(np.asarray(seg_im)[..., 0].copy())
                        obs_rew = self.reward_true_latent.sin().add_(1).mul_(0.5)
                        top_light_max_y = self.camera_raw_render_size // 3
                        for ii in range(3):
                            if ii == 0:
                                max_right = int(np.ceil(self.camera_raw_render_size * (35 / 120)))
                            elif ii == 1:
                                max_right = int(np.ceil(self.camera_raw_render_size * (50 / 120)))
                            elif ii == 2:
                                max_right = int(np.ceil(self.camera_raw_render_size * (65 / 120)))
                            rect = (slice(None, top_light_max_y), slice(None, max_right))
                            mask = np.isin(seg_im[rect], self.light_ids[ii])
                            v[rect][mask] = obs_rew[ii]

                    image_t = torch.stack((h, s, v), dim=-3)
                    img_hue_adj = _hsv2rgb(image_t)

                    image_t = (img_hue_adj * 255.0).to(dtype=torch.uint8)
                    image = image_t.permute(1, 2, 0).numpy()

                camera._scene.free()  # pylint: disable=protected-access

                pil_image = Image.fromarray(image).crop(box=self.VIEW_OFFSETS[view].get_cropbox(self.camera_raw_render_size))
                pil_image = pil_image.resize(
                    [self.image_size, self.image_size],
                    resample=self.resize_interpolation,
                )

                images.append(self.ndarray_uint8_image_to_observation(np.asarray(pil_image)))

            if len(images) == 1:
                return images[0]
            else:
                return torch.cat(images, dim=1).reshape(3 * len(self.views), *images[0].shape[1:])

    def sample_random_action(self, size=(), np_rng=None):
        if np_rng is None:
            np_rng = np.random
        return torch.as_tensor(np_rng.uniform(-1, 1, size=tuple(size) + tuple(self.action_shape)), dtype=torch.float32)

    @property
    def observation_output_kind(self) -> EnvBase.ObsOutputKind:
        return self._observation_output_kind

    @property
    def max_episode_length(self) -> int:
        return self.episode_length

    _action_repeat: int

    @property
    def action_repeat(self) -> int:
        return self._action_repeat

    @action_repeat.setter
    def action_repeat(self, value):
        self._action_repeat = value

    @property
    def observation_space(self):
        # if self.symbolic:
        #     return gym.spaces.Box(
        #         low=self.observation_output_kind.low,
        #         high=self.observation_output_kind.high,
        #         shape=[self._symbolic_state_size],
        #         dtype=self.observation_output_kind.np_dtype,
        #     )
        # else:
        return EnvBase.ObsOutputKind.get_observation_space(
            self.observation_output_kind, self.image_size, self.image_size,
            num_channels=3 * len(self.views))


def parse_RoboDeskEnv(env: str, observation_output_kind: EnvBase.ObsOutputKind, seed, max_episode_length,
                      action_repeat, batch_shape):
    if not env.startswith('robodesk_'):
        raise ValueError(env)

    terms = env.split('_')
    task_name = terms[1].replace('-', '_')
    terms = terms[2:]

    kwargs = dict(
        task=task_name,
        reward='dense',
        action_repeat=action_repeat, episode_length=max_episode_length,
        observation_output_kind=observation_output_kind,
        image_size=64,
    )

    tys = set()
    for term in terms:
        if term.startswith('button='):
            tyl = 6
            light = term[tyl + 1:]
            if light == 'noiseless':
                kwargs.update(
                    button_noise_scale=0,
                    button_noise_iid_scale=0,
                )
            elif light == 'Noisy':
                kwargs.update(
                    button_noise_scale=1.2,
                    button_noise_iid_scale=0.2,
                )
            elif light == 'Noisyy':
                kwargs.update(
                    button_noise_scale=0.2,
                    button_noise_smoothness=0.1,
                    button_noise_iid_scale=0.05,
                )
            elif light == 'NoisyyCorrelated' or light == 'NoistyCorrelated':  # BC typo
                kwargs.update(
                    button_noise_scale=0.295,
                    button_noise_smoothness=0.04,
                    button_noise_iid_scale=0.0125,
                )
            else:
                assert False
        elif term.startswith('objL='):
            tyl = 4
            light = term[tyl + 1:]
            if light == 'Bin':
                kwargs.update(
                    light_mechanism='new',
                )
            elif light == 'Range':
                kwargs.update(
                    light_mechanism='new+range',
                )
            elif light == 'RangeExt':
                kwargs.update(
                    light_mechanism='new+range+ext',
                )
            elif light == 'RangeExtOffset':
                kwargs.update(
                    light_mechanism='new+range+ext',
                    button_to_light_offset=0.78,
                )
            else:
                assert False
        elif term.startswith('rL='):
            tyl = 2
            spec = term[tyl + 1:]
            if spec == '':
                fns = []
            else:
                fns = [s.replace('-', '_') for s in spec.split(',')]
            kwargs.update(
                red_button_functions=fns,
            )
        elif term.startswith('gL='):
            tyl = 2
            spec = term[tyl + 1:]
            if spec == '':
                fns = []
            else:
                fns = [s.replace('-', '_') for s in spec.split(',')]
            kwargs.update(
                green_button_functions=fns,
            )
        elif term.startswith('bL='):
            tyl = 2
            spec = term[tyl + 1:]
            if spec == '':
                fns = []
            else:
                fns = [s.replace('-', '_') for s in spec.split(',')]
            kwargs.update(
                blue_button_functions=fns,
            )
        elif term.startswith('envL='):
            tyl = 4
            env_light = term[tyl + 1:]
            if env_light == 'Noiseless':
                kwargs.update(
                    env_light_swing_radius=0,
                    env_light_flicker_magnitude=0,
                    env_light_flicker_stddev=0,
                    env_light_swing_action_angle_speed=None,
                )
            elif env_light == 'Noisy':
                kwargs.update(
                    env_light_swing_angle_speed=[np.sqrt(5) / 15, 1 / 15, 1 / (15 * np.sqrt(3))],
                    # env_light_swing_radius=[0.25, 1 / np.sqrt(2), 1],
                    env_light_flicker_magnitude=[0.15, 0.15, 0.15],
                    env_light_flicker_stddev=[0.1, 0.1, 0.1],
                    env_light_swing_action_angle_speed=None,
                )
            elif env_light == 'NoisyAct':
                kwargs.update(
                    env_light_swing_angle_speed=[np.sqrt(5) / 15, 1 / 15, 1 / (15 * np.sqrt(3))],
                    # env_light_swing_radius=[0.25, 1 / np.sqrt(2), 1],
                    env_light_flicker_magnitude=[0.15, 0.15, 0.15],
                    env_light_flicker_stddev=[0.1, 0.1, 0.1],
                    env_light_swing_action_angle_speed=[0.05, 0.05, 0.05],
                )
            elif env_light == 'Noisyy':
                kwargs.update(
                    env_light_swing_angle_speed=[np.sqrt(5) / 15, 1 / 15, 1 / (15 * np.sqrt(3))],
                    env_light_swing_angle_stddev=[np.sqrt(5) / 30, 1 / 30, 1 / (30 * np.sqrt(3))],
                    env_light_flicker_magnitude=0.3,
                    env_light_flicker_stddev=[0.15, 0.15, 0.15],
                    env_light_flicker_base_brightness=-0.15,
                    env_light_swing_action_angle_speed=None,
                )
            elif env_light == 'NoisyyAct':
                kwargs.update(
                    env_light_swing_angle_speed=[np.sqrt(5) / 15, 1 / 15, 1 / (15 * np.sqrt(3))],
                    env_light_swing_angle_stddev=[np.sqrt(5) / 30, 1 / 30, 1 / (30 * np.sqrt(3))],
                    env_light_flicker_magnitude=0.3,
                    env_light_flicker_stddev=[0.15, 0.15, 0.15],
                    env_light_flicker_base_brightness=-0.15,
                    env_light_swing_action_angle_speed=[0.05, 0.05, 0.05],
                )
            else:
                assert False, env_light
        elif term.startswith('cam='):
            tyl = 3
            fromjitter, atjitter = term[tyl + 1:].split('>')
            kwargs.update(
                cam_lookfrom_jitter=float(fromjitter),
                cam_lookat_jitter=float(atjitter),
            )
        elif term.startswith('sliderAlph='):
            tyl = len('sliderAlph')
            kwargs.update(
                slider_panel_alpha=float(term[tyl + 1:]),
            )
        elif term.startswith('views='):
            tyl = 5
            kwargs.update(
                views=list(t.replace('-', '_') for t in term[tyl + 1:].split(',')),
            )
        elif term.startswith('track='):
            tyl = 5
            kwargs.update(
                track_info=list(t.replace('-', '_') for t in term[tyl + 1:].split(',')),
            )
        elif term.startswith('tv='):
            tyl = 2
            tv = term[tyl + 1:]
            tv_specs = tv.split('-')
            tvty = tv_specs[0]
            tv_texture_size = 120
            if tvty.startswith('Video2'):
                num_cols = 2
                tvty = tvty[6:]
            elif tvty.startswith('Video'):
                num_cols = 3
                tvty = tvty[5:]
            else:
                raise ValueError(tvty)
            assert tv_texture_size % num_cols == 0
            if tvty.endswith('DynContrast'):
                dynamic_contrast_sharpen = True
                tvty = tvty[:-11]
            else:
                dynamic_contrast_sharpen = False

            from ..utils import get_kinetics_dir
            if tvty == '':
                files = glob.glob(f'{get_kinetics_dir()}/train/driving_car/*.mp4')
                single_imagesource_fn = functools.partial(
                    natural_imgsource.RandomVideoSource,
                    (tv_texture_size // num_cols, tv_texture_size // num_cols),
                    files,
                    grayscale=False,
                    total_frames=1000,
                    interpolation=cv2.INTER_NEAREST,
                    dynamic_contrast_sharpen=dynamic_contrast_sharpen,
                )
            elif tvty == 'BW':
                files = glob.glob(f'{get_kinetics_dir()}/train/driving_car/*.mp4')
                single_imagesource_fn = functools.partial(
                    natural_imgsource.RandomVideoSource,
                    (tv_texture_size // num_cols, tv_texture_size // num_cols),
                    files,
                    grayscale=True,
                    total_frames=1000,
                    interpolation=cv2.INTER_NEAREST,
                    dynamic_contrast_sharpen=dynamic_contrast_sharpen,
                )
            elif tvty == 'SharpContrast':
                files = glob.glob(f'{get_kinetics_dir()}/train/driving_car/*.mp4')
                single_imagesource_fn = functools.partial(
                    natural_imgsource.RandomVideoSource,
                    (tv_texture_size // num_cols, tv_texture_size // num_cols),
                    files,
                    grayscale=False,
                    total_frames=1000,
                    interpolation=cv2.INTER_NEAREST,
                    contrast_delta=20,
                    sharpen=0.5,
                    dynamic_contrast_sharpen=dynamic_contrast_sharpen,
                )
            elif tvty == 'BWSharpContrast':
                files = glob.glob(f'{get_kinetics_dir()}/train/driving_car/*.mp4')
                single_imagesource_fn = functools.partial(
                    natural_imgsource.RandomVideoSource,
                    (tv_texture_size // num_cols, tv_texture_size // num_cols),
                    files,
                    grayscale=True,
                    total_frames=1000,
                    interpolation=cv2.INTER_NEAREST,
                    contrast_delta=20,
                    sharpen=0.5,
                    dynamic_contrast_sharpen=dynamic_contrast_sharpen,
                )
            elif tvty == 'Sharp':
                files = glob.glob(f'{get_kinetics_dir()}/train/driving_car/*.mp4')
                single_imagesource_fn = functools.partial(
                    natural_imgsource.RandomVideoSource,
                    (tv_texture_size // num_cols, tv_texture_size // num_cols),
                    files,
                    grayscale=False,
                    total_frames=1000,
                    interpolation=cv2.INTER_NEAREST,
                    sharpen=0.5,
                    dynamic_contrast_sharpen=dynamic_contrast_sharpen,
                )
            elif tvty == 'BWSharp':
                files = glob.glob(f'{get_kinetics_dir()}/train/driving_car/*.mp4')
                single_imagesource_fn = functools.partial(
                    natural_imgsource.RandomVideoSource,
                    (tv_texture_size // num_cols, tv_texture_size // num_cols),
                    files,
                    grayscale=True,
                    total_frames=1000,
                    interpolation=cv2.INTER_NEAREST,
                    sharpen=0.5,
                    dynamic_contrast_sharpen=dynamic_contrast_sharpen,
                )
            else:
                assert False, tv
            specs = dict(
                roll_offset_inc=0,
                frame_skip=1,
            )
            for tv_spec in tv_specs[1:]:
                if tv_spec.startswith('Roll'):
                    specs['roll_offset_inc'] = int(tv_spec[4:])
                elif tv_spec.startswith('FS'):
                    specs['frame_skip'] = int(tv_spec[2:])
                else:
                    assert False, tv_spec
            image_source_fn = utils.functools.compose(
                functools.partial(
                    natural_imgsource.ConcatImageSource,
                    axis=1,
                    roll_offset_inc=specs['roll_offset_inc'],
                ),  # List[ImageSource] -> ImageSource
                utils.functools.repeat(
                    single_imagesource_fn,
                    num_cols,
                ),
            )
            image_source_fn = utils.functools.compose(
                functools.partial(
                    natural_imgsource.FrameSkipImageSource,
                    frame_skip=specs['frame_skip'],
                ),
                image_source_fn,
            )
            kwargs.update(
                tv_image_source_fn=image_source_fn,
            )
        elif term.startswith('res='):
            tyl = 3
            kwargs.update(
                image_size=int(term[tyl + 1:]),
            )
        elif term.startswith('noiseV='):
            tyl = 6
            kwargs.update(
                noise_speed_factor=float(term[tyl + 1:]),
            )
        elif term.startswith('interp='):
            tyl = 6
            kwargs.update(
                resize_interpolation=getattr(Image, term[tyl + 1:].upper()),
            )
        elif term.startswith('variant='):
            tyl = 7
            kwargs.update(
                xml_file='desk_' + term[tyl + 1:].replace('-', '_') + '.xml',
            )
        else:
            assert False, term
        ty = term[:tyl]
        assert ty not in tys
        tys.add(ty)

    from ..utils import make_batched_auto_reset_env
    return make_batched_auto_reset_env(
        lambda seed: RoboDeskEnv(seed=seed, **kwargs),
        seed, batch_shape)


__all__ = ['parse_RoboDeskEnv', 'RoboDeskEnv']
