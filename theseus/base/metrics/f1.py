from typing import Any

from sklearn.metrics import f1_score

from theseus.base.metrics.metric_template import Metric
from theseus.base.utilities.logits import logits2labels


class F1ScoreMetric(Metric):
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
            return {f"{self.average}-f1": 0.0}
        score = f1_score(self.targets, self.preds, average=self.average)
        return {f"{self.average}-f1": score}

    def reset(self):
        self.targets = []
        self.preds = []
