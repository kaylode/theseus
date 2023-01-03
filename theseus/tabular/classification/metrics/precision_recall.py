from typing import Any, Dict

import numpy as np
from sklearn.metrics import precision_score, recall_score

from theseus.base.metrics.metric_template import Metric


class SKLPrecisionRecall(Metric):
    """
    F1 Score Metric (including macro, micro)
    """

    def __init__(self, average="weighted", **kwargs):
        super().__init__(**kwargs)
        self.average = average
        self.targets = []
        self.preds = []

    def value(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        targets = batch["targets"]
        outputs = outputs["outputs"]

        self.preds += np.argmax(outputs, axis=1).reshape(-1).tolist()
        self.targets += targets.reshape(-1).tolist()

        precision = precision_score(
            self.targets, self.preds, average=self.average, zero_division=1
        )
        recall = recall_score(
            self.targets, self.preds, average=self.average, zero_division=1
        )
        return {
            f"{self.average}-precision": precision,
            f"{self.average}-recall": recall,
        }
