from .base_callbacks import Callbacks, CallbacksList
from .logger_callbacks import LoggerCallbacks
from .checkpoint_callbacks import CheckpointCallbacks

from theseus.registry import Registry
CALLBACKS_REGISTRY = Registry('CALLBACKS')

CALLBACKS_REGISTRY.register(LoggerCallbacks)
CALLBACKS_REGISTRY.register(CheckpointCallbacks)