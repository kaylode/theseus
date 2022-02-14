from theseus.base.trainer import TRAINER_REGISTRY 

from .trainer import SegmentationTrainer

TRAINER_REGISTRY.register(SegmentationTrainer)