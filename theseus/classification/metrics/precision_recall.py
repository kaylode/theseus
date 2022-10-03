from sklearn.metrics import precision_score, recall_score
from typing import Any, Dict

from theseus.base.metrics.metric_template import Metric
from theseus.classification.utilities.logits import logits2labels

class PrecisionRecall(Metric):
    """
    F1 Score Metric (including macro, micro)
    """
    def __init__(self, average = 'weighted', label_type:str = 'multiclass', **kwargs):
        super().__init__(**kwargs)
        self.average = average
        self.type =label_type
        self.threshold = kwargs.get('threshold', 0.5)
        self.reset()

    def update(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        targets = batch["targets"] 
        outputs = outputs["outputs"] 

        outputs = logits2labels(outputs, label_type=self.type, threshold=self.threshold)
        targets = targets.squeeze()
    
        self.preds +=  outputs.numpy().tolist()
        self.targets +=  targets.numpy().tolist()

    def value(self):
        precision = precision_score(self.targets, self.preds, average=self.average, zero_division=1)
        recall = recall_score(self.targets, self.preds, average=self.average, zero_division=1)
        return {f"{self.average}-precision": precision, f"{self.average}-recall": recall}

    def reset(self):
        self.targets = []
        self.preds = []