from typing import Any, Dict

from theseus.base.metrics.metric_template import Metric
from theseus.base.utilities.logits import logits2labels


class Accuracy(Metric):
    """
    Accuracy metric
    """

    def __init__(self, label_type: str = "multiclass", ignore_index=None, **kwargs):
        super().__init__(**kwargs)
        self.type = label_type
        self.threshold = kwargs.get("threshold", 0.5)
        self.ignore_index = ignore_index
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

        # Create mask for non-ignored indices
        if self.ignore_index is not None:
            mask = target.view(-1) != self.ignore_index
            prediction = prediction.view(-1)[mask]
            target = target.view(-1)[mask]
        else:
            prediction = prediction.view(-1)
            target = target.view(-1)

        if len(target) > 0:  # Avoid division by zero if all targets are ignored
            correct = (prediction == target).sum()
            self.total_correct += correct
            self.sample_size += len(target)

    def value(self):
        if self.sample_size == 0:
            return {"acc": 0.0}
        return {"acc": (self.total_correct / self.sample_size).item()}

    def reset(self):
        self.total_correct = 0
        self.sample_size = 0
