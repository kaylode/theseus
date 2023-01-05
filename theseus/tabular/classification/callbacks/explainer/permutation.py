import os.path as osp
from typing import Dict, List

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance

from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


class PermutationImportance(Callbacks):
    def __init__(self, save_dir, **kwargs) -> None:
        super().__init__()
        self.explainer = None
        self.save_dir = save_dir

    def on_train_epoch_end(self, logs: Dict = None):
        """
        After finish training
        """
        model = self.params["trainer"].model.get_model()
        x_train, y_train = (
            logs["trainset"]["inputs"],
            logs["trainset"]["targets"],
        )
        feature_names = logs["trainset"]["feature_names"]
        classnames = logs["trainset"]["classnames"]

        perm_importance = permutation_importance(model, x_train, y_train)
        sorted_idx = perm_importance.importances_mean.argsort()

        fig = go.Figure(
            go.Bar(
                x=perm_importance.importances_mean[sorted_idx],
                y=feature_names[sorted_idx],
                orientation="h",
            )
        )

        plt.xlabel("Permutation Importance")
        save_path = osp.join(self.save_dir, "permutation_train.html")
        fig.write_html(save_path, auto_play = False)

        LOGGER.log(
            [
                {
                    "tag": f"Importance/permutation/train",
                    "value": save_path,
                    "type": LoggerObserver.HTML,
                }
            ]
        )

        LOGGER.text(
            f"Permutation figure saved at {save_path}",
            level=LoggerObserver.INFO,
        )
        plt.clf()

    def on_val_epoch_end(self, logs: Dict = None):
        """
        After finish validation
        """
        model = self.params["trainer"].model.get_model()
        x_val, y_val = logs["valset"]["inputs"], logs["valset"]["targets"]
        feature_names = logs["valset"]["feature_names"]
        classnames = logs["valset"]["classnames"]

        perm_importance = permutation_importance(model, x_val, y_val)
        sorted_idx = perm_importance.importances_mean.argsort()

        fig = go.Figure(
            go.Bar(
                x=perm_importance.importances_mean[sorted_idx],
                y=feature_names[sorted_idx],
                orientation="h",
            )
        )

        plt.xlabel("Permutation Importance")
        save_path = osp.join(self.save_dir, "permutation_val.html")
        fig.write_html(save_path, auto_play = False)

        LOGGER.log(
            [
                {
                    "tag": f"Importance/permutation/val",
                    "value": save_path,
                    "type": LoggerObserver.HTML,
                }
            ]
        )

        LOGGER.text(
            f"Permutation figure saved at {save_path}",
            level=LoggerObserver.INFO,
        )
        plt.clf()
