from theseus.base.trainer import TRAINER_REGISTRY

from .ml_trainer import MLTrainer

TRAINER_REGISTRY.register(MLTrainer)
