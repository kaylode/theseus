from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
)

# from theseus.ml.callbacks.base_callbacks import Callbacks, CallbacksList
from theseus.registry import Registry

from .checkpoint_callback import TorchCheckpointCallback
from .loss_logging_callback import LossLoggerCallback
from .metric_logging_callback import MetricLoggerCallback
from .timer_callback import TimerCallback
from .tsb_callback import TensorboardCallback
from .wandb_callback import WandbCallback

CALLBACKS_REGISTRY = Registry("CALLBACKS")

CALLBACKS_REGISTRY.register(TimerCallback)
CALLBACKS_REGISTRY.register(TensorboardCallback)
CALLBACKS_REGISTRY.register(WandbCallback)
CALLBACKS_REGISTRY.register(ModelCheckpoint)
CALLBACKS_REGISTRY.register(RichModelSummary)
CALLBACKS_REGISTRY.register(LearningRateMonitor)
CALLBACKS_REGISTRY.register(EarlyStopping)
CALLBACKS_REGISTRY.register(LossLoggerCallback)
CALLBACKS_REGISTRY.register(MetricLoggerCallback)
CALLBACKS_REGISTRY.register(TorchCheckpointCallback)
