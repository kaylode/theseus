from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.trainer import Trainer as plTrainer

from theseus.registry import Registry

TRAINER_REGISTRY = Registry("trainer")


class Trainer(plTrainer):
    def __init__(self, *args, **kwargs):
        if "use_nccl_strategy" in kwargs.keys() and "strategy" not in kwargs.keys():
            # Default to DDP strategy with NCCL backend
            use_nccl_strategy = kwargs.pop("use_nccl_strategy")
            if use_nccl_strategy:
                ddp = DDPStrategy(
                    process_group_backend="nccl", find_unused_parameters=True
                )
                super().__init__(strategy=ddp, *args, **kwargs)
            else:
                super().__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs)


TRAINER_REGISTRY.register(Trainer, prefix="pl")
