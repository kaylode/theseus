from theseus.registry import Registry

from .base_callbacks import Callbacks, CallbacksList
from .checkpoint_callbacks import TorchCheckpointCallbacks
from .debug_callbacks import DebugCallbacks
from .loss_logging_callbacks import LossLoggerCallbacks
from .metric_logging_callbacks import MetricLoggerCallbacks
from .timer_callbacks import TimerCallbacks
from .tsb_callbacks import TensorboardCallbacks
from .wandb_callbacks import WandbCallbacks

CALLBACKS_REGISTRY = Registry("CALLBACKS")

CALLBACKS_REGISTRY.register(TimerCallbacks)
CALLBACKS_REGISTRY.register(TorchCheckpointCallbacks)
CALLBACKS_REGISTRY.register(TensorboardCallbacks)
CALLBACKS_REGISTRY.register(WandbCallbacks)
CALLBACKS_REGISTRY.register(DebugCallbacks)
CALLBACKS_REGISTRY.register(LossLoggerCallbacks)
CALLBACKS_REGISTRY.register(MetricLoggerCallbacks)
