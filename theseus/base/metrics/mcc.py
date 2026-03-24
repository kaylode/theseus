from typing import Any

from sklearn.metrics import matthews_corrcoef

from theseus.base.metrics.metric_template import Metric
from theseus.base.utilities.logits import logits2labels


class MCC(Metric):
    """
    Mathew Correlation Coefficient
    """

    def __init__(self, label_type: str = "multiclass", ignore_index=None, **kwargs):
        super().__init__(**kwargs)
        self.type = label_type
        self.ignore_index = ignore_index
        self.reset()

    def update(self, outputs: dict[str, Any], batch: dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        targets = batch["targets"].cpu()
        outputs = outputs["outputs"].detach().cpu()
        outputs = logits2labels(outputs, label_type=self.type)

        # Filter out ignored indices
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            outputs = outputs[mask]
            targets = targets[mask]

        self.preds += outputs.numpy().tolist()
        self.targets += targets.numpy().tolist()

    def value(self):
        if len(self.targets) == 0:
            return {"mcc": 0.0}
        score = matthews_corrcoef(self.targets, self.preds)
        return {
            "mcc": score,
        }

    def reset(self):
        self.targets = []
        self.preds = []
