from typing import Dict, List

import lightning.pytorch as pl
import optuna
from lightning.pytorch.callbacks import Callback

from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


class OptunaCallback(Callback):
    """
    Callbacks for reporting value to optuna trials to decide whether to prune
    """

    def __init__(self, trial: optuna.Trial, **kwargs) -> None:
        super().__init__()
        self.trial = trial

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        After finish validation
        """

        iters = trainer.global_step
        metric_dict = pl_module.metric_dict

        best_key = self.trial.user_attrs["best_key"]
        self.trial.report(value=metric_dict[best_key], step=iters)

        if self.trial.should_prune():
            LOGGER.text(
                f"Trial {self.trial.number} has been pruned", level=LoggerObserver.DEBUG
            )
            raise optuna.TrialPruned()
