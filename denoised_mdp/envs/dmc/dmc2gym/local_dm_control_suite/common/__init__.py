# Copyright 2017 The dm_control Authors.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Functions to manage the common assets for domains."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from dm_control.utils import io as resources

_SUITE_DIR = os.path.dirname(os.path.dirname(__file__))
_FILENAMES = [
    "./common/materials.xml",
    "./common/materials_white_floor.xml",
    "./common/skybox.xml",
    "./common/visual.xml",
]

ASSETS = {filename: resources.GetResource(os.path.join(_SUITE_DIR, filename))
          for filename in _FILENAMES}


def read_model(model_filename):
  """Reads a model XML file and returns its contents as a string."""
  return resources.GetResource(os.path.join(_SUITE_DIR, model_filename))


import contextlib
from dm_control import mujoco

class NoisyMuJoCoPhysics(mujoco.Physics):
  SUPPORTS_NOISY_SENSOR = True
  NUM_NOISES = 0  # overwrite in subclasses

  noise_enabled: bool

  def set_noise_enabled(self, flag: bool):
    self.noise_enabled = flag

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.noise = None
    self.noise_enabled = False  # must explicitly enable via `set_noise_enabled`
    self.noise_type = None

  @contextlib.contextmanager
  def sensor_noise(self, noise):
    if not self.noise_enabled:
      assert noise is None, f"{self} does not enable noisy sensors"
    if noise is not None:
      assert isinstance(noise, (list, tuple)) and len(noise) == self.NUM_NOISES
    assert self.noise is None, 'cannot nest this context manager'
    self.noise = tuple(noise)  # shouldn't have any inplace changes, but be safe
    yield
    self.noise = None
