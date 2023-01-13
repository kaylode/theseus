import time
from typing import Dict, List

import numpy as np

from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


class LossLoggerCallbacks(Callbacks):
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

    def auto_get_print_interval(self, train_fraction=0.1):
        """
        Automatically decide the number of print interval
        """
        trainloader = self.params["trainer"].trainloader
        num_iterations_per_epoch = len(trainloader)
        print_interval = max(int(train_fraction * num_iterations_per_epoch), 1)
        return print_interval

    def on_start(self, logs: Dict = None):
        """
        Before going to the main loop
        """
        if self.print_interval is None:
            self.print_interval = self.auto_get_print_interval()
            LOGGER.text(
                "Print interval not specified. Auto calculating...",
                level=LoggerObserver.DEBUG,
            )

    def on_train_epoch_start(self, logs: Dict = None):
        """
        Before going to the training loop
        """
        self.running_loss = {}
        self.running_time_list = []

    def on_train_batch_start(self, logs: Dict = None):
        """
        Before going to the training loop
        """
        self.running_time = time.time()

    def on_train_batch_end(self, logs: Dict = None):
        """
        After finish a batch
        """

        lr = logs["lr"]
        iters = logs["iters"]
        loss_dict = logs["loss_dict"]
        num_iterations = logs["num_iterations"]
        trainloader_length = len(self.params["trainer"].trainloader)

        # Update running loss of batch
        for (key, value) in loss_dict.items():
            if key not in self.running_loss.keys():
                self.running_loss[key] = []
            self.running_loss[key].append(value)

        # Running time since last interval
        batch_time = time.time() - self.running_time
        self.running_time_list.append(batch_time)

        # Logging
        if iters % self.print_interval == 0 or (iters + 1) % trainloader_length == 0:

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
                    num_iterations,
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

    def on_val_epoch_start(self, logs: Dict = None):
        """
        Before main validation loops
        """
        self.running_time = time.time()
        self.running_loss = {}

    def on_val_batch_end(self, logs: Dict = None):
        """
        After finish a batch
        """

        loss_dict = logs["loss_dict"]

        # Update batch loss
        for (key, value) in loss_dict.items():
            if key not in self.running_loss.keys():
                self.running_loss[key] = []
            self.running_loss[key].append(value)

    def on_val_epoch_end(self, logs: Dict = None):
        """
        After finish validation
        """

        iters = logs["iters"]
        num_iterations = logs["num_iterations"]
        epoch_time = time.time() - self.running_time
        valloader = self.params["trainer"].valloader

        # Log loss
        for key in self.running_loss.keys():
            self.running_loss[key] = np.round(np.mean(self.running_loss[key]), 5)
        loss_string = (
            "{}".format(self.running_loss)[1:-1].replace("'", "").replace(",", " ||")
        )
        LOGGER.text(
            "[{}|{}] || {} || Time: {:10.4f} (it/s)".format(
                iters, num_iterations, loss_string, len(valloader) / epoch_time
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
