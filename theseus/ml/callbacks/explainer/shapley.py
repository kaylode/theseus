import os
import os.path as osp
from typing import Dict, List

import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance

from theseus.base.utilities.loggers.observer import LoggerObserver
from theseus.ml.callbacks import Callbacks

LOGGER = LoggerObserver.getLogger("main")


class ShapValueExplainer(Callbacks):
    def __init__(
        self, save_dir, plot_type="bar", check_additivity=True, **kwargs
    ) -> None:
        super().__init__()
        self.plot_type = plot_type
        self.explainer = None
        self.save_dir = save_dir
        self.check_additivity = check_additivity

    def on_train_epoch_end(self, logs: Dict = None):
        """
        After finish training
        """
        model = self.params["trainer"].model.get_model()
        self.explainer = shap.TreeExplainer(model)
        x_train, y_train = (
            logs["trainset"]["inputs"],
            logs["trainset"]["targets"],
        )
        feature_names = logs["trainset"]["feature_names"]
        classnames = logs["trainset"]["classnames"]
        shap_values = self.explainer.shap_values(
            x_train, check_additivity=self.check_additivity
        )
        shap.summary_plot(
            shap_values,
            plot_type=self.plot_type,
            feature_names=feature_names,
            class_names=classnames,
            show=False,
        )

        fig = plt.gcf()

        LOGGER.log(
            [
                {
                    "tag": "Importance/SHAP/train",
                    "value": fig,
                    "type": LoggerObserver.FIGURE,
                    "kwargs": {"step": 0},
                }
            ]
        )

        LOGGER.text(
            f"Shapley figure saved",
            level=LoggerObserver.INFO,
        )
        plt.clf()

    def on_validation_epoch_end(self, logs: Dict = None):
        """
        After finish validation
        """
        model = self.params["trainer"].model.get_model()
        self.explainer = shap.TreeExplainer(model)
        x_val, y_val = logs["valset"]["inputs"], logs["valset"]["targets"]
        feature_names = logs["valset"]["feature_names"]
        classnames = logs["valset"]["classnames"]
        shap_values = self.explainer.shap_values(
            x_val, check_additivity=self.check_additivity
        )
        plt.clf()
        shap.summary_plot(
            shap_values,
            plot_type=self.plot_type,
            feature_names=feature_names,
            class_names=classnames,
            show=False,
        )

        fig = plt.gcf()

        LOGGER.log(
            [
                {
                    "tag": "Importance/SHAP/val",
                    "value": fig,
                    "type": LoggerObserver.FIGURE,
                    "kwargs": {"step": 0},
                }
            ]
        )

        LOGGER.text(
            f"Shapley figure saved",
            level=LoggerObserver.INFO,
        )
        plt.clf()
