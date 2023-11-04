from typing import Any, Dict

import numpy as np
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score

from theseus.base.metrics.metric_template import Metric


class SKLAccuracy(Metric):
    """
    Accuracy metric
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def value(self, output: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        output = output["outputs"]
        target = batch["targets"]

        probs = softmax(output, axis=-1)
        predictions = np.argmax(probs, axis=-1)

        correct = (predictions.reshape(-1) == target.reshape(-1)).sum()
        score = correct * 1.0 / target.shape[0]
        return {"acc": score}


def compute_multiclass(outputs, targets, index):
    correct = 0
    sample_size = 0
    for i, j in zip(outputs, targets):
        if j == index:
            sample_size += 1
            if i == j:
                correct += 1
    return correct, sample_size


class SKLBalancedAccuracyMetric(Metric):
    """
    Balanced Accuracy metric for classification
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def value(self, outputs: Dict[str, Any], batch: Dict[str, Any]):

        outputs = outputs["outputs"]
        targets = batch["targets"]
        predictions = np.argmax(outputs, axis=-1).reshape(-1).tolist()
        targets = targets.reshape(-1).tolist()
        blacc_score = balanced_accuracy_score(targets, predictions)

        return {"bl_acc": blacc_score}
