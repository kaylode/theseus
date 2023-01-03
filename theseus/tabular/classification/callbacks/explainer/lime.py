import os.path as osp
import random
from typing import Dict, List

from lime import lime_tabular

from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


class LIMEExplainer(Callbacks):
    def __init__(self, save_dir, **kwargs) -> None:
        super().__init__()
        self.save_dir = save_dir

    def explain_instance(
        self, training_data, model, item, feature_names=None, class_names=None
    ):
        """
        Get explaination for a single instance
        """
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=class_names,
            mode="classification" if class_names is not None else "regression",
        )

        return self.explainer.explain_instance(
            data_row=item, predict_fn=model.predict_proba
        )

    def on_val_epoch_end(self, logs: Dict = None):
        """
        After finish validation
        """

        model = self.params["trainer"].model.get_model()
        x_train, y_train = (
            logs["trainset"]["inputs"],
            logs["trainset"]["targets"],
        )
        x_val, y_val = logs["valset"]["inputs"], logs["valset"]["targets"]
        feature_names = logs["valset"]["feature_names"]
        classnames = logs["valset"]["classnames"]

        item_id = random.choice(range(len(x_val)))
        item = x_val[item_id]
        exp = self.explain_instance(
            x_train,
            model,
            item,
            feature_names=feature_names,
            class_names=classnames,
        )
        save_path = osp.join(self.save_dir, f"lime_{item_id}.html")
        exp.save_to_file(save_path)
        LOGGER.text(
            f"LIME figure for a random instance saved at {save_path}",
            level=LoggerObserver.INFO,
        )
