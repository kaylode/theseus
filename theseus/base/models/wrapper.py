from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.amp import autocast

from theseus.base.datasets import LightningDataModuleWrapper
from theseus.base.optimizers import OPTIM_REGISTRY, SCHEDULER_REGISTRY
from theseus.base.utilities.getter import get_instance


class LightningModelWrapper(pl.LightningModule):
    """
    Lightning wrapper that bridges Theseus components to the Lightning training loop.

    Encapsulates model, criterion, optimizer, scheduler, and metrics into a single
    ``pl.LightningModule`` with proper step methods and automatic mixed precision.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module | None = None,
        *,
        metrics: list[Any] | None = None,
        optimizer_config: dict | None = None,
        scheduler_config: dict | None = None,
        scheduler_kwargs: dict | None = None,
        datamodule: LightningDataModuleWrapper | None = None,
        use_mixed_precision: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.scheduler_kwargs = scheduler_kwargs
        self.datamodule = datamodule
        self.use_mixed_precision = use_mixed_precision
        self.lr: float = 0.0
        self.metric_dict: dict[str, Any] = {}

    def log_dict(self, dictionary: Mapping[str, Any], **kwargs: Any) -> None:
        """Filter non-loggable values before passing to Lightning's log_dict."""
        filtered_dict = {
            key: value
            for key, value in dictionary.items()
            if isinstance(value, (torch.Tensor, float, int))
        }
        return super().log_dict(filtered_dict, **kwargs)

    @property
    def _autocast_device(self) -> str:
        """Detect the correct device type for autocast."""
        if self.device.type == "cuda":
            return "cuda"
        elif self.device.type == "mps":
            return "mps"
        return "cpu"

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # Use Lightning's optimizers() API instead of storing self.optimizer
        optimizers = self.optimizers()
        if optimizers is not None:
            opt = optimizers if not isinstance(optimizers, list) else optimizers[0]
            lrl = [x["lr"] for x in opt.param_groups]
            self.lr = sum(lrl) / len(lrl)

    def _compute_and_log_metrics(self, batch_size_key: str = "valloader") -> None:
        """Compute metrics and log them. Reduces duplication between val/test."""
        self.metric_dict = {}
        if self.metrics is not None:
            for metric in self.metrics:
                self.metric_dict.update(metric.value())
                metric.reset()

        loader = getattr(self.datamodule, batch_size_key, None)
        batch_size = loader.batch_size if loader is not None else 1
        self.log_dict(self.metric_dict, prog_bar=True, batch_size=batch_size)

    def on_validation_epoch_end(self) -> None:
        self._compute_and_log_metrics("valloader")

    def on_test_epoch_end(self) -> None:
        self._compute_and_log_metrics("testloader")

    def _forward(
        self,
        batch: dict[str, Any],
        metrics: list[Any] | None = None,
    ) -> dict[str, Any]:
        """
        Forward the batch through models, losses and metrics.
        If some parameters are needed, it's best to include in the batch.
        """
        device_type = self._autocast_device

        # BF16 is not supported on all devices, and autocast sometimes fails on CPU
        # if not explicitly supported. We'll be more conservative here.
        enabled = self.use_mixed_precision
        if device_type == "cpu" and enabled:
            # Most CPUs don't support BF16/FP16 well in autocast unless using specific CPUs
            # It's safer to disable for CPU unless it's explicitly managed by Lightning
            enabled = False

        with autocast(device_type=device_type, enabled=enabled):
            outputs = self.model.forward_batch(batch)
            if self.criterion is None:
                loss = outputs["outputs"].get("loss", None)
                loss_dict = outputs["outputs"].get("loss_dict", None)
                if loss is None or loss_dict is None:
                    raise ValueError(
                        "No loss found in model outputs. Please ensure the model returns a loss."
                    )
            else:
                loss, loss_dict = self.criterion(outputs, batch)

            if metrics is not None:
                for metric in metrics:
                    metric.update(outputs, batch)

        return {"loss": loss, "loss_dict": loss_dict, "model_outputs": outputs}

    def trainable_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        outputs = self._forward(batch)
        self.log_dict(outputs["loss_dict"], prog_bar=True, on_step=True, on_epoch=False)
        return outputs

    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        outputs = self._forward(batch, metrics=self.metrics)
        self.log_dict(outputs["loss_dict"], prog_bar=True, on_step=True, on_epoch=False)
        return outputs

    def test_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        outputs = self._forward(batch, metrics=self.metrics)
        self.log_dict(outputs["loss_dict"], prog_bar=True, on_step=True, on_epoch=False)
        return outputs

    def predict_step(self, batch: Any, batch_idx: int | None = None) -> Any:
        return self.model.get_prediction(batch)

    def configure_optimizers(self) -> Any:
        if self.optimizer_config is not None:
            self.optimizer = get_instance(
                self.optimizer_config,
                registry=OPTIM_REGISTRY,
                params=self.model.parameters(),
            )
        else:
            from torch.optim import AdamW

            self.optimizer = AdamW(self.parameters(), lr=self.lr)

        if self.scheduler_config is not None:
            self.scheduler = get_instance(
                self.scheduler_config,
                registry=SCHEDULER_REGISTRY,
                optimizer=self.optimizer,
                **self.scheduler_kwargs,
            )

            scheduler_interval = "epoch" if self.scheduler.step_per_epoch else "step"
            scheduler = {
                "scheduler": self.scheduler.scheduler,
                "interval": scheduler_interval,
            }
            return [self.optimizer], [scheduler]
        else:
            from torch.optim.lr_scheduler import LinearLR, SequentialLR

            n_steps = self.trainer.estimated_stepping_batches
            n_warmup_steps = int(0.1 * n_steps)
            n_decay_steps = int(0.9 * n_steps)

            warmup = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=n_warmup_steps,
            )
            decay = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=n_decay_steps,
            )
            self.scheduler = SequentialLR(
                optimizer=self.optimizer,
                schedulers=[warmup, decay],
                milestones=[n_warmup_steps],
            )

            scheduler = {
                "scheduler": self.scheduler,
                "interval": "step",
            }
            return [self.optimizer], [scheduler]
