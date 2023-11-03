from lightning.pytorch.trainer import Trainer

from theseus.registry import Registry

TRAINER_REGISTRY = Registry("trainer")

TRAINER_REGISTRY.register(Trainer, prefix="pl")
