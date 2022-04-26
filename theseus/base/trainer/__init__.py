from theseus.registry import Registry

from .base_trainer import BaseTrainer
from .supervised_trainer import SupervisedTrainer

TRAINER_REGISTRY = Registry('TRAINER')
TRAINER_REGISTRY.register(BaseTrainer)
TRAINER_REGISTRY.register(SupervisedTrainer)