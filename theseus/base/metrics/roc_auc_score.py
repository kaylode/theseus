from typing import Any, Dict

import torch
from scikitplot.metrics import plot_precision_recall_curve, plot_roc_curve
from sklearn.metrics import roc_auc_score

from theseus.base.metrics.metric_template import Metric
from theseus.base.utilities.cuda import detach, move_to


class ROCAUCScore(Metric):
    """
    Area Under Curve, ROC Curve Score
    """

    def __init__(self, average: str = "weighted", label_type: str = "ovr", **kwargs):
        super().__init__(**kwargs)
        self.label_type = label_type
        self.average = average
        assert self.label_type in [
            "raise",
            "ovr",
            "ovo",
        ], "Invalid type for multiclass ROC AUC score"
        self.reset()

    def update(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        targets = batch["targets"]
        outputs = move_to(outputs["outputs"], torch.device("cpu"))

        if self.label_type == "ovr":
            outputs = torch.softmax(outputs, dim=-1)

        self.preds += outputs.numpy().tolist()
        self.targets += targets.numpy().tolist()

    def value(self):
        roc_auc_scr = roc_auc_score(
            self.targets, self.preds, average=self.average, multi_class=self.label_type
        )
        roc_curve_fig = plot_roc_curve(self.targets, self.preds).get_figure()
        pr_fig = plot_precision_recall_curve(self.targets, self.preds).get_figure()

        return {
            f"{self.average}-roc_auc_score": roc_auc_scr,
            "roc_curve": roc_curve_fig,
            "precision_recall_curve": pr_fig,
        }

    def reset(self):
        self.targets = []
        self.preds = []
