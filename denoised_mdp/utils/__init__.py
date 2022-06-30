# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import os
import shutil
import functools


def rm_if_exists(filename, maybe_dir=False) -> bool:
    r"""
    Returns whether removed
    """
    if os.path.isfile(filename):
        os.remove(filename)
        return True
    elif maybe_dir and os.path.isdir(filename):
        shutil.rmtree(filename)
        return True
    assert not os.path.exists(filename)
    return False


T = TypeVar('T')


class lazy_property(Generic[T]):
    r"""
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    Derived from:
      https://github.com/pytorch/pytorch/blob/556c8a300b5b062f3429dfac46f6def372bd22fc/torch/distributions/utils.py#L92
    TODO: replace with `functools.cached_property` in py3.8.
    """

    def __init__(self, wrapped: Callable[[Any], T]):
        self.wrapped = wrapped
        functools.update_wrapper(self, wrapped)

    def __get__(self, instance: Any, obj_type: Any = None) -> T:
        if instance is None:
            return self  # typing: ignore
        value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value


from . import logging


__all__ = ['rm_if_exists', 'lazy_property', 'logging']