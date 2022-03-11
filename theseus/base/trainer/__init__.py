from theseus.registry import Registry
from .base_trainer import BaseTrainer
from .supervised_trainer import SupervisedTrainer

TRAINER_REGISTRY = Registry('TRAINER')
TRAINER_REGISTRY.register(BaseTrainer)
TRAINER_REGISTRY.register(SupervisedTrainer)

from theseus.base.trainer.callbacks import LoggerCallbacks, CheckpointCallbacks

TRAINER_REGISTRY.register(LoggerCallbacks)
TRAINER_REGISTRY.register(CheckpointCallbacks)