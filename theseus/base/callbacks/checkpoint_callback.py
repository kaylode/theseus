import os
import os.path as osp
import inspect
import lightning.pytorch as pl
from lightning.pytorch.callbacks import  ModelCheckpoint
from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


class TorchCheckpointCallback(ModelCheckpoint):
    def __init__(self, save_dir: str, **kwargs) -> None:

        save_dir = osp.join(save_dir, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        inspection = inspect.signature(ModelCheckpoint)
        class_kwargs = inspection.parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in class_kwargs}

        super().__init__(
            dirpath=save_dir, 
            **filtered_kwargs
        )

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        self.params = {}
        trainloader = pl_module.datamodule.trainloader
        if trainloader is not None:
            batch_size = trainloader.batch_size
            self.params['trainloader_length'] = len(trainloader)
            self.params['num_iterations'] = len(trainloader) * batch_size * trainer.max_epochs

        if self._every_n_train_steps is None or self._every_n_train_steps == 0:
            LOGGER.text("Save interval not specified. Auto calculating...", level=LoggerObserver.DEBUG)
            self._every_n_train_steps = self.auto_get_save_interval()

    def auto_get_save_interval(self, train_fraction=0.5):
        """
        Automatically decide the number of save interval
        """
        save_interval = max(int(train_fraction * self.params['trainloader_length']), 1)
        return save_interval
    
    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)

        if filepath in self.best_k_models.keys():
            if self.best_k_models[filepath] == self.best_model_score:
                LOGGER.text(
                    f"Evaluation improved to {self.current_score}",
                    level=LoggerObserver.SUCCESS,
                )

        LOGGER.text(
            f"Save checkpoints to {filepath}",
            level=LoggerObserver.INFO,
        )
