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

from .base.utilities import (
    LoggerObserver,
    download_from_wandb,
    find_file_recursively,
    get_devices_info,
    get_instance_recursively,
    move_to,
    seed_everything,
)
from .registry import Registry

__all__ = [
    "move_to",
    "seed_everything",
    "LoggerObserver",
    "find_file_recursively",
    "get_devices_info",
    "download_from_wandb",
    "get_instance_recursively",
    "Registry",
]
