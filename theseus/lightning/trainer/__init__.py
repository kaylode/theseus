from theseus.base.trainer import TRAINER_REGISTRY
from lightning.pytorch.trainer import Trainer

TRAINER_REGISTRY.register(Trainer, prefix="pl")