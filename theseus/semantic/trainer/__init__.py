from theseus.base.trainer import TRAINER_REGISTRY 

from .trainer import SemanticTrainer

TRAINER_REGISTRY.register(SemanticTrainer)