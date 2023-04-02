import time
from typing import Dict, List

import numpy as np

from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


class AutoFindLRCallbacks(Callbacks):
    """
    Callbacks for auto finding LR
    :params:
    lr_range: List
        learning rate search space
    gamma: int
        number of iterations per lr step
    """

    def __init__(
        self, lr_range: List[float], num_steps: int, num_epochs: int = 1, **kwargs
    ) -> None:
        super().__init__()

        self.lr_range = lr_range
        self.num_steps = num_steps
        self.num_epochs = num_epochs

        assert (
            self.lr_range[1] > self.lr_range[0]
        ), "Learning rate range should be from low to high"
        assert self.num_epochs > 0, "Num epochs should be higher than 0"

    def auto_get_interval(self):
        """
        Automatically decide the number of interval
        """
        trainloader = self.params["trainer"].trainloader
        num_iterations = len(trainloader) * self.num_epochs

        num_iterations_per_steps = (num_iterations - 1) // self.num_steps
        step_iters = [
            int(round(x * num_iterations_per_steps)) for x in range(0, self.num_steps)
        ]

        gamma = (self.lr_range[1] - self.lr_range[0]) / float(self.num_steps - 1)
        lrs = [self.lr_range[0] + x * gamma for x in range(0, self.num_steps)]

        return step_iters, lrs

    def on_start(self, logs: Dict = None):
        """
        Before going to the main loop
        """

        LOGGER.text(
            "Autofinding LR is activated. Running for 1 epoch only...",
            level=LoggerObserver.DEBUG,
        )

        trainloader = self.params["trainer"].trainloader
        num_iterations = len(trainloader) * self.num_epochs
        self.params["trainer"].num_iterations = num_iterations

        self.step_iters, self.lrs = self.auto_get_interval()
        self.current_idx = 0
        LOGGER.text(
            "Interval for Learning Rate AutoFinding not specified. Auto calculating...",
            level=LoggerObserver.DEBUG,
        )

        self.tracking_loss = []
        self.tracking_lr = []

        optim = self.params["trainer"].optimizer
        for g in optim.param_groups:
            g["lr"] = self.lrs[self.current_idx]
        self.current_idx += 1

    def on_train_batch_end(self, logs: Dict = None):
        """
        After finish a batch
        """

        lr = logs["lr"]
        iters = logs["iters"]
        loss_dict = logs["loss_dict"]
        optim = self.params["trainer"].optimizer

        log_dict = [
            {
                "tag": f"AutoLR/{k} Loss",
                "value": v,
                "type": LoggerObserver.SCALAR,
                "kwargs": {"step": iters},
            }
            for k, v in loss_dict.items()
        ]

        # Log learning rates
        log_dict.append(
            {
                "tag": "AutoLR/Learning rate",
                "value": lr,
                "type": LoggerObserver.SCALAR,
                "kwargs": {"step": iters},
            }
        )

        LOGGER.log(log_dict)

        self.tracking_loss.append(sum([v for v in loss_dict.values()]))
        self.tracking_lr.append(lr)

        # Logging
        if (
            self.current_idx < len(self.step_iters)
            and iters == self.step_iters[self.current_idx]
        ):
            for g in optim.param_groups:
                g["lr"] = self.lrs[self.current_idx]
            self.current_idx += 1
