"""
Theseus - A modular PyTorch Lightning framework for deep learning
~~~~~~~~~~~~~~~~~~~

:copyright: (c) 2020-present Kaylode
:license: MIT, see LICENSE for more details.

"""

__title__ = "theseus"
__author__ = "kaylode"
__license__ = "MIT"
__copyright__ = "Copyright 2020-present Kaylode"
__version__ = "2.0.0"

from .base import *
from .registry import Registry

from .base.pipeline import BasePipeline
from .base.datasets.sampler import BalanceSampler
from .base.datasets.dataloader import DataLoaderWithCollator

from .base.utilities import (
    move_to,
    seed_everything,
    LoggerObserver,
    find_file_recursively,
    get_devices_info,
    download_from_wandb,
    get_instance_recursively,
)  # explicitly re-exporting

from .base.utilities.loggers import FileLogger

__all__ = [
    "move_to",
    "seed_everything",
    "FileLogger",
    "LoggerObserver",
    "find_file_recursively",
    "get_devices_info",
    "download_from_wandb",
    "get_instance_recursively",
    "BasePipeline",
    "BalanceSampler",
    "DataLoaderWithCollator",
    "Registry",
]