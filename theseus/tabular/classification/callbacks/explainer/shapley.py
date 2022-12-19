from typing import Dict, List
import os
import os.path as osp
import shap
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")

class ShapValueExplainer(Callbacks):
    def __init__(self, save_dir, plot_type='bar', **kwargs) -> None:
        super().__init__()
        self.plot_type = plot_type
        self.explainer = None
        self.save_dir = save_dir

    def on_train_epoch_end(self, logs: Dict=None):
        """
        After finish training
        """
        model = self.params['trainer'].model.get_model()
        self.explainer = shap.TreeExplainer(model)
        x_train, y_train = logs['trainset']['inputs'], logs['trainset']['targets']
        feature_names = logs['trainset']['feature_names']
        classnames = logs['trainset']['classnames']
        shap_values = self.explainer.shap_values(x_train)
        shap.summary_plot(
            shap_values, 
            plot_type=self.plot_type, 
            feature_names=feature_names,
            class_names=classnames,
            show=False
        )

        save_path = osp.join(self.save_dir, 'shapley_train')
        plt.savefig(save_path)
        LOGGER.text(f"Shapley figure saved at {save_path+'.jpg'}", level=LoggerObserver.INFO)
        plt.clf()

    def on_val_epoch_end(self, logs: Dict=None):
        """
        After finish validation
        """
        model = self.params['trainer'].model.get_model()
        self.explainer = shap.TreeExplainer(model)
        x_val, y_val = logs['valset']['inputs'], logs['valset']['targets']
        feature_names = logs['valset']['feature_names']
        classnames = logs['valset']['classnames']
        shap_values = self.explainer.shap_values(x_val)
        shap.summary_plot(
            shap_values, 
            plot_type=self.plot_type, 
            feature_names=feature_names,
            class_names=classnames,
            show=False
        )

        save_path = osp.join(self.save_dir, 'shapley_val')
        plt.savefig(save_path)
        LOGGER.text(f"Shapley figure saved at {save_path+'.jpg'}", level=LoggerObserver.INFO)
        plt.clf()
