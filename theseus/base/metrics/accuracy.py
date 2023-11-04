from typing import Any, Dict

from theseus.base.metrics.metric_template import Metric
from theseus.base.utilities.logits import logits2labels


class Accuracy(Metric):
    """
    Accuracy metric
    """

    def __init__(self, label_type: str = "multiclass", **kwargs):
        super().__init__(**kwargs)
        self.type = label_type
        self.threshold = kwargs.get("threshold", 0.5)
        self.reset()

    def update(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        outputs = outputs["outputs"].detach().cpu()
        target = batch["targets"].cpu()
        prediction = logits2labels(
            outputs, label_type=self.type, threshold=self.threshold
        )

        correct = (prediction.view(-1) == target.view(-1)).sum()

        self.total_correct += correct
        self.sample_size += prediction.view(-1).size(0)

    def value(self):
        return {"acc": (self.total_correct / self.sample_size).item()}

    def reset(self):
        self.total_correct = 0
        self.sample_size = 0
