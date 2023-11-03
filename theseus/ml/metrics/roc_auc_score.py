from typing import Any, Dict

import numpy as np
import scipy

from theseus.base.metrics.metric_template import Metric

try:
    from scikitplot.metrics import plot_precision_recall_curve, plot_roc_curve

    has_scikitplot = True
except:
    has_scikitplot = False
from sklearn.metrics import roc_auc_score


class SKLROCAUCScore(Metric):
    """
    ROC AUC Score
    """

    def __init__(
        self,
        average: str = "weighted",
        label_type: str = "ovr",
        plot_curve: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.plot_curve = plot_curve
        self.label_type = label_type
        self.average = average
        assert self.label_type in [
            "raise",
            "ovr",
            "ovo",
        ], "Invalid type for multiclass ROC AUC score"

    def value(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        targets = batch["targets"]
        outputs = outputs["outputs"]

        if self.label_type == "ovr":
            outputs = scipy.special.softmax(outputs, axis=-1)

        self.preds = outputs.tolist()
        self.targets = targets.reshape(-1).tolist()

        roc_auc_scr = roc_auc_score(
            self.targets, self.preds, average=self.average, multi_class=self.label_type
        )
        results = {
            f"{self.average}-roc_auc_score": roc_auc_scr,
        }

        if has_scikitplot and self.plot_curve:
            roc_curve_fig = plot_roc_curve(self.targets, self.preds).get_figure()
            pr_fig = plot_precision_recall_curve(self.targets, self.preds).get_figure()
            results.update(
                {
                    "roc_curve": roc_curve_fig,
                    "precision_recall_curve": pr_fig,
                }
            )

        return results
