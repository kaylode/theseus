import time
from typing import Dict, List, Any
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")

class LossLoggerCallback(Callback):
    """
    Callbacks for logging running loss while training
    Features:
        - Only do logging

    print_interval: `int`
        iteration cycle to log out
    """

    def __init__(self, print_interval: int = None, **kwargs) -> None:
        super().__init__()

        self.running_time = 0
        self.running_loss = {}
        self.print_interval = print_interval

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """
        Setup the callback
        """
        self.params = {}


        trainloader = pl_module.datamodule.trainloader
        if trainloader is not None:
            batch_size = trainloader.batch_size
            self.params['num_iterations'] = len(trainloader) * batch_size * trainer.max_epochs
            self.params['trainloader_length'] = len(trainloader)
        else:
            self.params['num_iterations'] = None
            self.params['trainloader_length'] = None

        valloader = pl_module.datamodule.valloader
        if valloader is not None:
            batch_size = valloader.batch_size
            self.params['valloader_length'] = len(valloader)
        else:
            self.params['valloader_length'] = None

        if self.print_interval is None:
            self.print_interval = self.auto_get_print_interval(pl_module)
            LOGGER.text(
                "Print interval not specified. Auto calculating...",
                level=LoggerObserver.DEBUG,
            )
            
    def auto_get_print_interval(self, pl_module: pl.LightningModule, train_fraction:float=0.1):
        """
        Automatically decide the number of print interval
        """

        num_iterations_per_epoch = self.params['trainloader_length'] if self.params['trainloader_length'] is not None else self.params['valloader_length']
        print_interval = max(int(train_fraction * num_iterations_per_epoch), 1)
        return print_interval
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Before going to the training loop
        """
        self.running_loss = {}
        self.running_time_list = []

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int):
        """
        Before going to the training loop
        """
        self.running_time = time.time()

    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int
        ):
        """
        After finish a batch
        """

        lr = pl_module.lr
        iters = trainer.global_step
        loss_dict = outputs['loss_dict']

        # Update running loss of batch
        for (key, value) in loss_dict.items():
            if key not in self.running_loss.keys():
                self.running_loss[key] = []
            self.running_loss[key].append(value)

        # Running time since last interval
        batch_time = time.time() - self.running_time
        self.running_time_list.append(batch_time)

        # Logging
        if iters % self.print_interval == 0 or (iters + 1) % self.params['trainloader_length'] == 0:

            # Running loss since last interval
            for key in self.running_loss.keys():
                self.running_loss[key] = np.round(np.mean(self.running_loss[key]), 5)
            loss_string = (
                "{}".format(self.running_loss)[1:-1]
                .replace("'", "")
                .replace(",", " ||")
            )

            # Running time average
            running_time = 1.0 / np.round(np.mean(self.running_time_list), 5)

            LOGGER.text(
                "[{}|{}] || {} || Time: {:10.4f} (it/s)".format(
                    iters,
                    self.params['num_iterations'],
                    loss_string,
                    running_time,
                ),
                LoggerObserver.INFO,
            )

            log_dict = [
                {
                    "tag": f"Training/{k} Loss",
                    "value": v,
                    "type": LoggerObserver.SCALAR,
                    "kwargs": {"step": iters},
                }
                for k, v in self.running_loss.items()
            ]

            # Log batch time execution
            log_dict.append(
                {
                    "tag": f"Training/Iterations per second",
                    "value": running_time,
                    "type": LoggerObserver.SCALAR,
                    "kwargs": {"step": iters},
                }
            )

            # Log learning rates
            log_dict.append(
                {
                    "tag": "Training/Learning rate",
                    "value": lr,
                    "type": LoggerObserver.SCALAR,
                    "kwargs": {"step": iters},
                }
            )

            LOGGER.log(log_dict)

            # Start new interval
            self.running_loss = {}
            self.running_time_list = []

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Before main validation loops
        """
        self.running_time = time.time()
        self.running_loss = {}

    def on_validation_batch_end(
            self, 
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: STEP_OUTPUT | None,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0
        ):
        """
        After finish a batch
        """

        loss_dict = outputs["loss_dict"]

        # Update batch loss
        for (key, value) in loss_dict.items():
            if key not in self.running_loss.keys():
                self.running_loss[key] = []
            self.running_loss[key].append(value)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        After finish validation
        """

        iters = trainer.global_step
        num_iterations = self.params["num_iterations"]
        epoch_time = time.time() - self.running_time

        # Log loss
        for key in self.running_loss.keys():
            self.running_loss[key] = np.round(np.mean(self.running_loss[key]), 5)
        loss_string = (
            "{}".format(self.running_loss)[1:-1].replace("'", "").replace(",", " ||")
        )
        LOGGER.text(
            "[{}|{}] || {} || Time: {:10.4f} (it/s)".format(
                iters, num_iterations, loss_string, self.params['valloader_length'] / epoch_time
            ),
            level=LoggerObserver.INFO,
        )

        # Call other loggers
        log_dict = [
            {
                "tag": f"Validation/{k} Loss",
                "value": v,
                "type": LoggerObserver.SCALAR,
                "kwargs": {"step": iters},
            }
            for k, v in self.running_loss.items()
        ]

        LOGGER.log(log_dict)