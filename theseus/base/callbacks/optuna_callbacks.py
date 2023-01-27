from typing import Dict, List

import optuna

from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


class OptunaCallbacks(Callbacks):
    """
    Callbacks for reporting value to optuna trials to decide whether to prune
    """

    def __init__(self, trial: optuna.Trial, **kwargs) -> None:
        super().__init__()
        self.trial = trial

    def on_val_epoch_end(self, logs: Dict = None):
        """
        After finish validation
        """

        iters = logs["iters"]
        metric_dict = logs["metric_dict"]

        best_key = self.trial.user_attrs["best_key"]
        self.trial.report(value=metric_dict[best_key], step=iters)

        if self.trial.should_prune():
            LOGGER.text(
                f"Trial {self.trial.number} has been pruned", level=LoggerObserver.DEBUG
            )
            raise optuna.TrialPruned()
