from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import _METRIC, STEP_OUTPUT

from theseus.base.datasets import LightningDataModuleWrapper
from theseus.base.optimizers import OPTIM_REGISTRY, SCHEDULER_REGISTRY
from theseus.base.utilities.getter import get_instance


class LightningModelWrapper(pl.LightningModule):
    """
    Wrapper for Lightning Module
    Instansiates the model, criterion, optimizer and scheduler
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = None,
        metrics: List[Any] = None,
        optimizer_config: Dict = None,
        scheduler_config: Dict = None,
        scheduler_kwargs: Dict = None,
        datamodule: LightningDataModuleWrapper = None,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.scheduler_kwargs = scheduler_kwargs
        self.datamodule = datamodule
        self.lr = 0
        self.metric_dict = {}

    def log_dict(self, dictionary: Mapping[str, Any], **kwargs) -> None:
        filtered_dict = {
            key: value
            for key, value in dictionary.items()
            if isinstance(value, (torch.Tensor, float, int))
        }
        return super().log_dict(filtered_dict, **kwargs)

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        lrl = [x["lr"] for x in self.optimizer.param_groups]
        self.lr = sum(lrl) / len(lrl)

    def on_validation_epoch_end(self) -> None:
        self.metric_dict = {}
        if self.metrics is not None:
            for metric in self.metrics:
                self.metric_dict.update(metric.value())
                metric.reset()

        self.log_dict(
            self.metric_dict,
            prog_bar=True,
            batch_size=self.datamodule.valloader.batch_size,
        )

    def on_test_epoch_end(self) -> None:
        self.metric_dict = {}
        if self.metrics is not None:
            for metric in self.metrics:
                self.metric_dict.update(metric.value())
                metric.reset()

        self.log_dict(
            self.metric_dict,
            prog_bar=True,
            batch_size=self.datamodule.testloader.batch_size,
        )

    def _forward(self, batch: Dict, metrics: List[Any] = None):
        """
        Forward the batch through models, losses and metrics
        If some parameters are needed, it's best to include in the batch
        """

        outputs = self.model.forward_batch(batch)
        loss, loss_dict = self.criterion(outputs, batch)

        if metrics is not None:
            for metric in metrics:
                metric.update(outputs, batch)

        return {"loss": loss, "loss_dict": loss_dict, "model_outputs": outputs}

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        outputs = self._forward(batch)
        self.log_dict(outputs["loss_dict"], prog_bar=True, on_step=True, on_epoch=False)
        return outputs

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        outputs = self._forward(batch, metrics=self.metrics)
        self.log_dict(outputs["loss_dict"], prog_bar=True, on_step=True, on_epoch=False)
        return outputs

    def test_step(self, batch, batch_idx):
        # this is the test loop
        outputs = self._forward(batch, metrics=self.metrics)
        self.log_dict(outputs["loss_dict"], prog_bar=True, on_step=True, on_epoch=False)
        return outputs

    def predict_step(self, batch, batch_idx=None):
        pred = self.model.get_prediction(batch)
        return pred

    def configure_optimizers(self):
        if self.optimizer_config is not None:
            self.optimizer = get_instance(
                self.optimizer_config,
                registry=OPTIM_REGISTRY,
                params=self.model.parameters(),
            )

        if self.scheduler_config is not None:
            self.scheduler = get_instance(
                self.scheduler_config,
                registry=SCHEDULER_REGISTRY,
                optimizer=self.optimizer,
                **self.scheduler_kwargs,
            )
        else:
            return self.optimizer

        scheduler_interval = "epoch" if self.scheduler.step_per_epoch else "step"
        scheduler = {
            "scheduler": self.scheduler.scheduler,
            "interval": scheduler_interval,
        }
        return [self.optimizer], [scheduler]
