from typing import Any

import torch

try:
    from scikitplot.metrics import plot_precision_recall, plot_roc

    has_scikitplot = True
except:
    has_scikitplot = False
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from theseus.base.metrics.metric_template import Metric
from theseus.base.utilities.cuda import move_to
from theseus.base.utilities.loggers.observer import LoggerObserver
from theseus.base.utilities.logits import logits2labels

LOGGER = LoggerObserver.getLogger("main")


class ROCAUCScore(Metric):
    """
    Area Under Curve, ROC Curve Score
    """

    def __init__(
        self,
        average: str = "weighted",
        label_type: str = "multiclass",
        plot_curve: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = label_type
        self.average = average
        self.plot_curve = plot_curve

        if self.type == "multiclass" or self.type == "multilabel":
            self.label_type = "ovr"
        else:
            self.label_type = "raise"

        self.reset()

    def update(self, outputs: dict[str, Any], batch: dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        targets = batch["targets"].cpu()
        outputs = move_to(outputs["outputs"], torch.device("cpu"))

        if self.type == "multiclass":
            probs = torch.softmax(outputs, dim=1)
            self.preds.extend(probs.numpy().tolist())
        else:
            _, probs = logits2labels(outputs, label_type=self.type, return_probs=True)
            self.preds += probs.numpy().tolist()
        self.targets += targets.view(-1).numpy().tolist()

    def value(self):
        try:
            roc_auc_scr = roc_auc_score(
                self.targets,
                self.preds,
                average=self.average,
                multi_class=self.label_type,
            )
        except Exception:
            try:
                preds = np.array(self.preds)
                if preds.ndim == 2:
                    preds = preds[:, 1]
                roc_auc_scr = roc_auc_score(
                    self.targets,
                    preds,
                    average=self.average,
                    multi_class=self.label_type,
                )
            except Exception as e:
                LOGGER.text(f"AUC score could not be calculated: {e}", level=LoggerObserver.WARN)
                roc_auc_scr = 0

        try:
            precision, recall, _ = precision_recall_curve(self.targets, self.preds, pos_label=1)
            pr_auc = auc(recall, precision)
        except Exception:
            try:
                preds = np.array(self.preds)
                if preds.ndim == 2:
                    preds = preds[:, 1]
                precision, recall, _ = precision_recall_curve(self.targets, preds, pos_label=1)
                pr_auc = auc(recall, precision)
            except Exception as e:
                LOGGER.text(
                    f"Precision-Recall AUC score could not be calculated: {e}",
                    level=LoggerObserver.WARN,
                )
                pr_auc = 0

        results = {
            f"{self.average}-roc_auc_score": roc_auc_scr,
            "pr_auc_score": pr_auc,
        }
        if has_scikitplot and self.plot_curve:
            roc_curve_fig = plot_roc(self.targets, self.preds).get_figure()
            pr_fig = plot_precision_recall(self.targets, self.preds).get_figure()
            results.update(
                {
                    "roc_curve": roc_curve_fig,
                    "precision_recall_curve": pr_fig,
                }
            )

        return results

    def reset(self):
        self.targets = []
        self.preds = []
