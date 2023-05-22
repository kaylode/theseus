from theseus.base.callbacks import CALLBACKS_REGISTRY
from lightning.pytorch.callbacks import (
    TQDMProgressBar, GradientAccumulationScheduler,
    RichProgressBar, BatchSizeFinder, ModelCheckpoint, OnExceptionCheckpoint, 
    RichModelSummary
)

from .wrapper import LightningCallbackWrapper, convert_to_lightning_callbacks

CALLBACKS_REGISTRY.register(TQDMProgressBar, prefix="pl")
CALLBACKS_REGISTRY.register(RichProgressBar, prefix="pl")
CALLBACKS_REGISTRY.register(GradientAccumulationScheduler, prefix="pl")
CALLBACKS_REGISTRY.register(BatchSizeFinder, prefix="pl")
CALLBACKS_REGISTRY.register(ModelCheckpoint, prefix="pl")
CALLBACKS_REGISTRY.register(OnExceptionCheckpoint, prefix="pl")
CALLBACKS_REGISTRY.register(RichModelSummary, prefix="pl")


try:
    from finetuning_scheduler import FinetuningScheduler
    CALLBACKS_REGISTRY.register(FinetuningScheduler, prefix="pl")
except ImportError:
    pass

