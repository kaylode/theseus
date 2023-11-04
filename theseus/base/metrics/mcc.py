from typing import Any, Dict

from sklearn.metrics import matthews_corrcoef

from theseus.base.metrics.metric_template import Metric
from theseus.base.utilities.logits import logits2labels


class MCC(Metric):
    """
    Mathew Correlation Coefficient
    """

    def __init__(self, label_type: str = "multiclass", **kwargs):
        super().__init__(**kwargs)
        self.type = label_type
        self.reset()

    def update(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        targets = batch["targets"].cpu()
        outputs = outputs["outputs"].detach().cpu()
        outputs = logits2labels(outputs, label_type=self.type)

        self.preds += outputs.numpy().tolist()
        self.targets += targets.numpy().tolist()

    def value(self):
        score = matthews_corrcoef(self.targets, self.preds)
        return {
            f"mcc": score,
        }

    def reset(self):
        self.targets = []
        self.preds = []
