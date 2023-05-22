from typing import List, Any
import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks import Callback as LightningCallback
from theseus.base.callbacks import Callbacks as TheseusCallbacks


def convert_to_lightning_callbacks(callbacks: List[TheseusCallbacks]) -> List[LightningCallback]:
    return [LightningCallbackWrapper(callback) if isinstance(callback, TheseusCallbacks) else callback for callback in callbacks]

class LightningCallbackWrapper(LightningCallback):
    """Wrapper for Lightning Callbacks to be used in Theseus
    https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.Callback.html
    """

    shared_memory: dict = {}

    def __init__(self, callback: TheseusCallbacks):
        self.callback = callback

    def _create_trainer_config(
            self, 
            params
        ) -> None:
        class ParamDict:
            def __init__(self, params: dict):
                for key, value in params.items():
                    setattr(self, key, value)

        return ParamDict(params)
    
    def on_sanity_check_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        
        trainloader = trainer.datamodule.trainloader
        batch_size = trainloader.batch_size
        self.num_iterations = len(trainloader) * batch_size
        placeholder_dict = self._create_trainer_config({
            'trainloader': trainer.datamodule.trainloader,
            'valloader': trainer.datamodule.valloader,
            'num_iterations' : self.num_iterations
        })
        self.callback.set_params({
            'trainer': placeholder_dict,
        })

        if getattr(self.callback, 'sanitycheck', None):
            self.callback.sanitycheck(logs={
                'iters': pl_module.iterations,
                'num_iterations': placeholder_dict.num_iterations
            })

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.callback.params is None:
            trainloader = trainer.datamodule.trainloader
            self.num_iterations = len(trainloader) * trainer.max_epochs
            placeholder_dict = self._create_trainer_config({
                'trainloader': trainer.datamodule.trainloader,
                'valloader': trainer.datamodule.valloader,
                'num_iterations' : self.num_iterations
            })
            self.callback.set_params({
                'trainer': placeholder_dict,
            })

        if getattr(self.callback, 'on_start', None):
            self.callback.on_start(logs={})
    
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if getattr(self.callback, 'on_finish', None):
            self.callback.on_finish(logs={
                'iters': pl_module.iterations,
                'num_iterations': self.num_iterations
            })

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if getattr(self.callback, 'on_epoch_start', None):
            self.callback.on_epoch_start(logs={
                'iters': pl_module.iterations,
            })

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if getattr(self.callback, 'on_train_epoch_start', None):
            self.callback.on_train_epoch_start(logs={})

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if getattr(self.callback, 'on_train_epoch_end', None):
            self.callback.on_train_epoch_end(logs={
                'last_batch': self.shared_memory['last_batch']
            })

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int) -> None:
        if getattr(self.callback, 'on_train_batch_start', None):
            self.callback.on_train_batch_start(logs={})

    def on_train_batch_end(
        self, 
        trainer: pl.Trainer, pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any, batch_idx: int
    ) -> None:
    
        if getattr(self.callback, 'on_train_batch_end', None):
            self.callback.on_train_batch_end(logs={
                'iters': pl_module.iterations,
                'loss_dict': outputs['loss_dict'],
                'lr': pl_module.lr,
                'num_iterations': self.num_iterations,
            })
        self.shared_memory['last_batch'] = batch

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if getattr(self.callback, 'on_val_epoch_start', None):
            self.callback.on_val_epoch_start(logs={})

    def on_validation_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int) -> None:
        if getattr(self.callback, 'on_val_batch_start', None):
            self.callback.on_val_batch_start(logs={'batch': batch})
        self.shared_memory['last_batch'] = batch

    def on_validation_batch_end(
        self, 
        trainer: pl.Trainer, pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any, batch_idx: int
    ) -> None:
        if getattr(self.callback, 'on_val_batch_end', None):
            self.callback.on_val_batch_end(logs={
                'iters': pl_module.iterations,
                'loss_dict': outputs['loss_dict'],
                'last_outputs': outputs['model_outputs'],
            })
        self.shared_memory['last_outputs'] = outputs['model_outputs']

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if getattr(self.callback, 'on_val_epoch_end', None):
            self.callback.on_val_epoch_end(logs={
                'iters': pl_module.iterations,
                'last_batch': self.shared_memory['last_batch'],
                'last_outputs': self.shared_memory['last_outputs'],
                'metric_dict': pl_module.metric_dict,
                "num_iterations": self.num_iterations,
            })