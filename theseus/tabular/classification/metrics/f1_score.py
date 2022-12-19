from sklearn.metrics import f1_score
from typing import Any, Dict
import numpy as np
from theseus.base.metrics.metric_template import Metric

class SKLF1ScoreMetric(Metric):
    """
    F1 Score Metric (including macro, micro)
    """
    def __init__(self, average = 'weighted', **kwargs):
        super().__init__(**kwargs)
        self.average = average
        self.preds = []
        self.targets = []

    def value(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        targets = batch["targets"] 
        outputs = outputs["outputs"] 

        self.preds += np.argmax(outputs,axis=1).reshape(-1).tolist()
        self.targets += targets.reshape(-1).tolist()
    
        score = f1_score(self.targets, self.preds, average=self.average)
        return {f"{self.average}-f1": score}