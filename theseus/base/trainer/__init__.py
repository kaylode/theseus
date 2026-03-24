from __future__ import annotations

from typing import Any

from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.trainer import Trainer as plTrainer

from theseus.registry import Registry

TRAINER_REGISTRY = Registry("trainer")


class Trainer(plTrainer):
    """
    Extended Lightning Trainer with convenience options for distributed strategies.

    Supports ``use_nccl_strategy`` for auto-configuring DDP with NCCL backend,
    and ``use_fsdp_strategy`` for Fully Sharded Data Parallel.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Handle NCCL DDP strategy shortcut
        use_nccl = kwargs.pop("use_nccl_strategy", False)
        use_fsdp = kwargs.pop("use_fsdp_strategy", False)

        if "strategy" not in kwargs:
            if use_nccl:
                kwargs["strategy"] = DDPStrategy(
                    process_group_backend="nccl",
                    find_unused_parameters=True,
                )
            elif use_fsdp:
                try:
                    from lightning.pytorch.strategies import FSDPStrategy

                    kwargs["strategy"] = FSDPStrategy()
                except ImportError:
                    raise ImportError(
                        "FSDP strategy requires lightning >= 2.0. "
                        "Please upgrade your lightning installation."
                    )

        super().__init__(*args, **kwargs)


TRAINER_REGISTRY.register(Trainer, prefix="pl")