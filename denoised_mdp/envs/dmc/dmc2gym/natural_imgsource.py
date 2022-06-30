# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# See also denoised_mdp/envs/dmc/dmc2gym/README.md for license-related
# information about these files adapted from
# https://github.com/facebookresearch/deep_bisim4control/

from typing import *

import abc
import numpy as np
import cv2
import skvideo.io
import tqdm

from .... import utils


class BackgroundMatting(object):
    """
    Produce a mask by masking the given color. This is a simple strategy
    but effective for many games.
    """
    def __init__(self, color):
        """
        Args:
            color: a (r, g, b) tuple or single value for grayscale
        """
        self._color = color

    def get_mask(self, img):
        return img == self._color


class ImageSource(abc.ABC):
    """
    Source of natural images to be added to a simulated environment.
    """

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, ...]:
        pass

    @abc.abstractmethod
    def get_image(self) -> np.ndarray:
        """
        Returns:
            an RGB image of [h, w, 3] with a fixed shape.
        """
        pass

    def increment(self, amount=1):
        pass

    def reset(self):
        """ Called when an episode ends. """
        pass

    def seed(self, seed: Union[int, np.random.SeedSequence]):
        pass

    def get_random_state(self) -> Any:
        pass

    def set_random_state(self, random_state):
        pass


class ConcatImageSource(ImageSource):
    def __init__(self, sources: List[ImageSource], axis=1, roll_offset_inc=0):
        assert len(sources) > 0
        self.sources = sources
        self.axis = axis
        assert roll_offset_inc >= 0
        self.roll_offset_inc = roll_offset_inc
        self.roll_offset = 0

    @utils.lazy_property
    def shape(self) -> Tuple[int, ...]:
        shape = list(self.sources[0].shape)
        for src in self.sources[1:]:
            assert all(d == self.axis or shape[d] == src.shape[d] for d in range(len(src.shape)))
            shape[self.axis] += src.shape[self.axis]
        return tuple(shape)

    def get_image(self, **kwargs):
        per_source_kwargs = [{} for _ in self.sources]
        for k, v in kwargs.items():
            if isinstance(v, (tuple, list)):
                assert len(v) == len(self.sources)
            else:
                v = [v for _ in self.sources]
            for ii in range(len(self.sources)):
                per_source_kwargs[ii][k] = v[ii]
        image: np.ndarray = np.concatenate(
            [s.get_image(**per_source_kwargs[ii]) for ii, s in enumerate(self.sources)],
            axis=self.axis,
        )
        if self.roll_offset != 0:
            image = np.roll(image, axis=self.axis, shift=int(np.round(self.roll_offset)))
        return image

    def increment(self, amount=1):
        self.roll_offset += self.roll_offset_inc * amount
        for s in self.sources:
            s.increment(amount=amount)

    def reset(self):
        self.roll_offset = 0
        for s in self.sources:
            s.reset()

    def seed(self, seed: Union[int, np.random.SeedSequence]):
        from ...utils import as_SeedSequence
        sss = as_SeedSequence(seed).spawn(len(self.sources))
        for s, ss in zip(self.sources, sss):
            s.seed(ss)

    def get_random_state(self) -> Any:
        return [s.get_random_state() for s in self.sources]

    def set_random_state(self, random_state):
        for s, rs in zip(self.sources, random_state):
            s.set_random_state(rs)


class FrameSkipImageSource(ImageSource):
    def __init__(self, source: ImageSource, frame_skip: int):
        self.source = source
        assert frame_skip > 0
        self.frame_skip = frame_skip

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.source.shape

    def get_image(self, **kwargs):
        return self.source.get_image(**kwargs)

    def increment(self, amount=1):
        return self.source.increment(amount=amount * self.frame_skip)

    def reset(self):
        return self.source.reset()

    def seed(self, seed: Union[int, np.random.SeedSequence]):
        return self.source.seed(seed)

    def get_random_state(self) -> Any:
        return self.source.get_random_state()

    def set_random_state(self, random_state):
        return self.source.set_random_state(random_state)


class FixedColorSource(ImageSource):
    def __init__(self, shape, color):
        """
        Args:
            shape: [h, w]
            color: a 3-tuple
        """
        self.arr = np.zeros((shape[0], shape[1], 3))
        self.arr[:, :] = color

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.arr.shape)

    def get_image(self):
        return self.arr


class RandomColorSource(ImageSource):
    def __init__(self, shape, seed=None):
        """
        Args:
            shape: [h, w]
        """
        self.hw_shape = shape
        self.arr = None
        self.np_random = np.random.Generator(np.random.PCG64(seed))
        self.reset()

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.arr.shape)

    def reset(self):
        self._color = self.np_random.integers(0, 256, size=(3,))
        self.arr = np.zeros((self.hw_shape[0], self.hw_shape[1], 3))
        self.arr[:, :] = self._color

    def get_image(self):
        return self.arr

    def get_random_state(self) -> Any:
        return self.np_random.bit_generator.state

    def set_random_state(self, random_state):
        self.np_random.bit_generator.state = random_state


class NoiseSource(ImageSource):
    def __init__(self, shape, strength=255, seed=None):
        """
        Args:
            shape: [h, w]
            strength (int): the strength of noise, in range [0, 255]
        """
        self.hw_shape = shape
        self.strength = strength
        self.np_random = np.random.Generator(np.random.PCG64(seed))

    @utils.lazy_property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.hw_shape) + (3,)

    def get_image(self):
        return self.np_random.normal(size=(self.hw_shape[0], self.hw_shape[1], 3)) * self.strength

    def seed(self, seed):
        self.np_random = np.random.Generator(np.random.PCG64(seed))

    def get_random_state(self) -> Any:
        return self.np_random.bit_generator.state

    def set_random_state(self, random_state):
        self.np_random.bit_generator.state = random_state


class RandomImageSource(ImageSource):
    def __init__(self, shape, filelist, total_frames=None, grayscale=False, seed=None):
        """
        Args:
            shape: [h, w]
            filelist: a list of image files
        """
        self.grayscale = grayscale
        self.hw_shape = shape
        self.filelist = filelist
        self.requested_total_frames = total_frames

        self._arr_built = False
        self.shuffle_random = np.random.Generator(np.random.PCG64())
        self.np_random = np.random.Generator(np.random.PCG64())
        self.seed(seed)
        self.reset()

    @utils.lazy_property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.hw_shape) + ((3,) if not self.grayscale else (1,),)

    def build_arr_if_needed(self):
        if self._arr_built:
            return
        self.total_frames = self.requested_total_frames if self.requested_total_frames else len(self.filelist)
        self.arr = np.zeros((self.total_frames, self.hw_shape[0], self.hw_shape[1]) + ((3,) if not self.grayscale else (1,)))
        for i in range(self.total_frames):
            if i % len(self.filelist) == 0:
                self.shuffle_random.shuffle(self.filelist)
            fname = self.filelist[i % len(self.filelist)]
            if self.grayscale:
                im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)[..., None]
            else:
                im = cv2.imread(fname, cv2.IMREAD_COLOR)
            self.arr[i] = cv2.resize(im, (self.hw_shape[1], self.hw_shape[0]))  # THIS IS NOT A BUG! cv2 uses (width, height)
        self._arr_built = True

    def seed(self, seed: Union[int, np.random.SeedSequence, None]):
        from ...utils import as_SeedSequence
        shuffle_rng_ss, np_ss = as_SeedSequence(seed).spawn(2)
        self.set_random_state((
            np.random.PCG64(shuffle_rng_ss).state,
            np.random.PCG64(np_ss).state,
        ))

    def get_random_state(self) -> Any:
        # (random_state_for_building_arr [either current candidiate or actually used], np_rng_state)
        return self._random_state_to_build_arr, self.np_random.bit_generator.state

    def set_random_state(self, random_state):
        random_state_to_build_arr = random_state[0]
        if self._arr_built:
            assert random_state_to_build_arr == self._random_state_to_build_arr
        else:
            self._random_state_to_build_arr = random_state_to_build_arr
            self.shuffle_random.bit_generator.state = random_state_to_build_arr
        self.np_random.bit_generator.state = random_state[1]

    def reset(self):
        self._need_actual_reset = True

    def get_image(self):
        self.build_arr_if_needed()
        if self._need_actual_reset:
            self._loc = self.np_random.integers(0, self.total_frames)
            self._need_actual_reset = False
        return self.arr[self._loc]

    def increment(self, amount=1):
        self._loc += amount


class RandomVideoSource(ImageSource):
    def __init__(self, shape, filelist, total_frames=None, grayscale=False, seed=None,
                 interpolation=cv2.INTER_LINEAR, contrast_delta=0, sharpen=0,
                 dynamic_contrast_sharpen=False):
        """
        Args:
            shape: [h, w]
            filelist: a list of video files
        """
        self.grayscale = grayscale
        self.hw_shape = shape
        self.filelist = filelist
        self.requested_total_frames = total_frames
        self.interpolation = interpolation
        self.contrast_delta = contrast_delta
        self.sharpen = sharpen
        self.dynamic_contrast_sharpen = dynamic_contrast_sharpen

        self._arr_built = False
        self.shuffle_random = np.random.Generator(np.random.PCG64())
        self.np_random = np.random.Generator(np.random.PCG64())
        self.seed(seed)
        self.reset()

    @utils.lazy_property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.hw_shape) + (1 if self.grayscale else 3,)

    def seed(self, seed: Union[int, np.random.SeedSequence, None]):
        from ...utils import as_SeedSequence
        shuffle_rng_ss, np_ss = as_SeedSequence(seed).spawn(2)
        self.set_random_state((
            np.random.PCG64(shuffle_rng_ss).state,
            np.random.PCG64(np_ss).state,
        ))

    def get_random_state(self) -> Any:
        # (random_state_for_building_arr [either current candidiate or actually used], np_rng_state)
        return self._random_state_to_build_arr, self.np_random.bit_generator.state

    def set_random_state(self, random_state):
        random_state_to_build_arr = random_state[0]
        if self._arr_built:
            assert random_state_to_build_arr == self._random_state_to_build_arr
        else:
            self._random_state_to_build_arr = random_state_to_build_arr
            self.shuffle_random.bit_generator.state = random_state_to_build_arr
        self.np_random.bit_generator.state = random_state[1]

    @staticmethod
    def manipulate_contrast_sharpen(frame, contrast_delta=0, sharpen=0):
        grayscale = frame.ndim == 2

        if contrast_delta != 0:
            contrast_delta = max(-127, min(127, float(contrast_delta)))
            f = 131 * (contrast_delta + 127) / (127 * (131 - contrast_delta))
            alpha_c = f
            gamma_c = 127 * (1 - f)
            frame = cv2.addWeighted(frame, alpha_c, frame, 0, gamma_c)

        if sharpen != 0:
            s = float(sharpen)
            kernel = np.array([
                [0, -s, 0],
                [-s, 1 + 4 * s, -s],
                [0, -s, 0],
            ])
            frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)

        if grayscale and frame.ndim == 2:
            frame = frame[..., None]  # cv2 strips the single dim for some reasons

        return frame

    def process_frame(self, frame):
        frame = cv2.resize(frame, (self.hw_shape[1], self.hw_shape[0]), interpolation=self.interpolation)  # THIS IS NOT A BUG! cv2 uses (width, height)
        if self.grayscale:
            frame = frame[..., None]
        if not self.dynamic_contrast_sharpen:
            frame = self.manipulate_contrast_sharpen(frame, contrast_delta=self.contrast_delta, sharpen=self.sharpen)
        return frame

    def build_arr_if_needed(self):
        if self._arr_built:
            return
        if not self.requested_total_frames:
            self.total_frames = 0
            self.arr = None
            self.shuffle_random.shuffle(self.filelist)
            for fname in tqdm.tqdm(self.filelist, desc="Loading videos for natural", position=0):
                if self.grayscale:
                    frames = skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})
                else:
                    frames = skvideo.io.vread(fname)
                local_arr = np.zeros((frames.shape[0], self.hw_shape[0], self.hw_shape[1]) + ((3,) if not self.grayscale else (1,)), dtype=np.uint8)
                for i in tqdm.tqdm(range(frames.shape[0]), desc="video frames", position=1):
                    local_arr[i] = self.process_frame(frames[i])
                if self.arr is None:
                    self.arr = local_arr
                else:
                    self.arr = np.concatenate([self.arr, local_arr], 0)
                self.total_frames += local_arr.shape[0]
        else:
            self.total_frames = self.requested_total_frames
            self.arr = np.zeros((self.total_frames, self.hw_shape[0], self.hw_shape[1]) + ((3,) if not self.grayscale else (1,)), dtype=np.uint8)
            total_frame_i = 0
            file_i = 0
            with tqdm.tqdm(total=self.total_frames, desc="Loading videos for natural") as pbar:
                while total_frame_i < self.total_frames:
                    if file_i % len(self.filelist) == 0:
                        self.shuffle_random.shuffle(self.filelist)
                    file_i += 1
                    fname = self.filelist[file_i % len(self.filelist)]
                    if self.grayscale:
                        frames = skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})
                    else:
                        frames = skvideo.io.vread(fname)
                    for frame_i in range(frames.shape[0]):
                        if total_frame_i >= self.total_frames:
                            break
                        self.arr[total_frame_i] = self.process_frame(frames[frame_i])
                        pbar.update(1)
                        total_frame_i += 1
        self._arr_built = True

    def reset(self):
        self._need_actual_reset = True
        self._inc_after_actual_reset = 0

    def get_image(self, contrast_delta=None, contrast_delta_delta=None, sharpen=None):
        self.build_arr_if_needed()
        if self._need_actual_reset:
            self._loc = self.np_random.integers(0, self.total_frames) + self._inc_after_actual_reset
            self._need_actual_reset = False
        frame = self.arr[int(np.round(self._loc)) % self.total_frames]

        if not self.dynamic_contrast_sharpen:
            assert contrast_delta is None
            assert contrast_delta_delta is None
            assert sharpen is None
        else:
            if self.grayscale:
                frame = frame[..., 0]
            if contrast_delta_delta is not None:
                assert contrast_delta is None
                contrast_delta = self.contrast_delta + contrast_delta_delta
                contrast_delta_delta = None
            if contrast_delta is None:
                contrast_delta = self.contrast_delta
            else:
                assert contrast_delta_delta is None
            if sharpen is None:
                sharpen = self.sharpen
            frame = self.manipulate_contrast_sharpen(frame, contrast_delta=contrast_delta, sharpen=sharpen)
        return frame

    def increment(self, amount=1):
        if self._need_actual_reset:
            self._inc_after_actual_reset += amount
        else:
            self._loc += amount
