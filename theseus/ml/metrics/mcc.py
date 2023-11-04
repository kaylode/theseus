from typing import Any, Dict

import numpy as np
from sklearn.metrics import matthews_corrcoef

from theseus.base.metrics.metric_template import Metric


class SKLMCC(Metric):
    """
    Mathew Correlation Coefficient
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def value(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        targets = batch["targets"]
        outputs = outputs["outputs"]

        self.preds = np.argmax(outputs, axis=1).reshape(-1).tolist()
        self.targets = targets.reshape(-1).tolist()

        score = matthews_corrcoef(self.targets, self.preds)
        return {f"mcc": score}
