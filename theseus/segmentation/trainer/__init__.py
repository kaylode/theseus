from theseus.base.trainer import TRAINER_REGISTRY 

from .trainer import ClassificationTrainer

TRAINER_REGISTRY.register(ClassificationTrainer)