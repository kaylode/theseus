from torckay.registry import Registry

TRAINER_REGISTRY = Registry('TRAINER')

from .base_trainer import BaseTrainer
from .supervised_trainer import SupervisedTrainer
