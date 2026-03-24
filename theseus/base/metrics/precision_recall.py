from typing import Any

from sklearn.metrics import precision_score, recall_score

from theseus.base.metrics.metric_template import Metric
from theseus.base.utilities.logits import logits2labels


class PrecisionRecall(Metric):
    """
    F1 Score Metric (including macro, micro)
    """

    def __init__(
        self, average="weighted", label_type: str = "multiclass", ignore_index=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.average = average
        self.type = label_type
        self.threshold = kwargs.get("threshold", 0.5)
        self.ignore_index = ignore_index
        self.reset()

    def update(self, outputs: dict[str, Any], batch: dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        targets = batch["targets"].cpu().view(-1)
        outputs = outputs["outputs"].detach().cpu()
        outputs = logits2labels(outputs, label_type=self.type, threshold=self.threshold)

        # Filter out ignored indices
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            outputs = outputs[mask]
            targets = targets[mask]

        self.preds += outputs.numpy().tolist()
        self.targets += targets.numpy().tolist()

    def value(self):
        if len(self.targets) == 0:
            return {
                f"{self.average}-precision": 0.0,
                f"{self.average}-recall": 0.0,
            }

        precision = precision_score(self.targets, self.preds, average=self.average, zero_division=1)
        recall = recall_score(self.targets, self.preds, average=self.average, zero_division=1)
        return {
            f"{self.average}-precision": precision,
            f"{self.average}-recall": recall,
        }

    def reset(self):
        self.targets = []
        self.preds = []
