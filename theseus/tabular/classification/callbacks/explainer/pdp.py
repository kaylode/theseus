import os.path as osp
from typing import Dict, List

import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


class PartialDependencePlots(Callbacks):
    def __init__(
        self,
        save_dir,
        feature_names,
        target_name,
        kind="both",
        num_jobs=1,
        num_samples=50,
        figsize=(10, 12),
        **kwargs,
    ) -> None:

        super().__init__()
        self.feature_names = feature_names
        self.kind = kind
        self.save_dir = save_dir
        self.num_jobs = num_jobs
        self.num_samples = num_samples
        self.figsize = figsize
        self.target_name = target_name

        self.num_cols = int(len(feature_names) / 3) + 1
        self.num_rows = int(len(feature_names) / self.num_cols)

    def on_train_epoch_end(self, logs: Dict = None):
        """
        After finish training
        """
        model = self.params["trainer"].model.get_model()
        x_train, y_train = (
            logs["trainset"]["inputs"],
            logs["trainset"]["targets"],
        )
        all_feature_names = logs["trainset"]["feature_names"]

        fig, ax = plt.subplots(self.num_rows, self.num_cols, figsize=self.figsize)
        PartialDependenceDisplay.from_estimator(
            model,
            x_train,
            self.feature_names,
            feature_names=all_feature_names,
            target=self.target_name,
            ax=ax,
            n_jobs=self.num_jobs,
            n_cols=self.num_cols,
            subsample=self.num_samples,
        )
        fig.suptitle("Partial Dependence Plots")
        fig.tight_layout()

        LOGGER.log(
            [
                {
                    "tag": "Importance/PDP/train",
                    "value": fig,
                    "type": LoggerObserver.FIGURE,
                    'kwargs': {"step": 0}
                }
            ]
        )

        LOGGER.text(
            f"PDP figure saved",
            level=LoggerObserver.INFO,
        )
        plt.clf()

    def on_val_epoch_end(self, logs: Dict = None):
        """
        After finish validation
        """
        model = self.params["trainer"].model.get_model()
        x_val, y_val = logs["valset"]["inputs"], logs["valset"]["targets"]
        all_feature_names = logs["valset"]["feature_names"]

        fig, ax = plt.subplots(self.num_rows, self.num_cols, figsize=self.figsize)
        PartialDependenceDisplay.from_estimator(
            model,
            x_val,
            self.feature_names,
            feature_names=all_feature_names,
            target=self.target_name,
            ax=ax,
            n_jobs=self.num_jobs,
            n_cols=self.num_cols,
            subsample=self.num_samples,
        )
        fig.suptitle("Partial Dependence Plots")
        fig.tight_layout()
        
        LOGGER.log(
            [
                {
                    "tag": "Importance/PDP/val",
                    "value": fig,
                    "type": LoggerObserver.FIGURE,
                    'kwargs': {"step": 0}
                }
            ]
        )

        LOGGER.text(
            f"PDP figure saved",
            level=LoggerObserver.INFO,
        )
        plt.clf()
