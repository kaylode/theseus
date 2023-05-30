from theseus.registry import Registry
from lightning.pytorch.trainer import Trainer

TRAINER_REGISTRY = Registry("trainer")

TRAINER_REGISTRY.register(Trainer, prefix="pl")