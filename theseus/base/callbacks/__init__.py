from theseus.registry import Registry

from .base_callbacks import Callbacks, CallbacksList
from .checkpoint_callbacks import CheckpointCallbacks
from .logger_callbacks import LoggerCallbacks
from .tsb_callbacks import TensorboardCallbacks
from .wandb_callbacks import WandbCallbacks

CALLBACKS_REGISTRY = Registry("CALLBACKS")

CALLBACKS_REGISTRY.register(LoggerCallbacks)
CALLBACKS_REGISTRY.register(CheckpointCallbacks)
CALLBACKS_REGISTRY.register(TensorboardCallbacks)
CALLBACKS_REGISTRY.register(WandbCallbacks)
