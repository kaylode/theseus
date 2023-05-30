import time
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


def seconds_to_hours(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    s = round(s, 4)
    return h, m, s


class TimerCallback(Callback):
    """
    Callbacks for logging running loss/metric/time while training
    Features:
        - Only do logging
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.running_time = 0
        self.start_time = 0

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Before going to the main loop
        """
        self.start_time = time.time()
        LOGGER.text(
            f"===========================START TRAINING=================================",
            level=LoggerObserver.INFO,
        )

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        After the main loop
        """
        LOGGER.text("Training Completed!", level=LoggerObserver.INFO)
        running_time = time.time() - self.start_time

        h, m, s = seconds_to_hours(running_time)
        LOGGER.text(
            f"Total running time: {h} hours, {m} minutes and {s} seconds",
            level=LoggerObserver.INFO,
        )

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Before going to the training loop
        """
        self.train_epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        After going to the training loop
        """
        running_time = time.time() - self.train_epoch_start_time
        h, m, s = seconds_to_hours(running_time)
        LOGGER.text(
            f"Training epoch running time: {h} hours, {m} minutes and {s} seconds",
            level=LoggerObserver.INFO,
        )

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Before main validation loops
        """
        self.val_epoch_start_time = time.time()
        LOGGER.text(
            "=============================EVALUATION===================================",
            LoggerObserver.INFO,
        )

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        After finish validation
        """

        running_time = time.time() - self.val_epoch_start_time
        h, m, s = seconds_to_hours(running_time)
        LOGGER.text(
            f"Evaluation epoch running time: {h} hours, {m} minutes and {s} seconds",
            level=LoggerObserver.INFO,
        )
        LOGGER.text(
            "================================================================",
            LoggerObserver.INFO,
        )
