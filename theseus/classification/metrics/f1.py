import torch
import numpy as np
from sklearn.metrics import f1_score
from typing import Any, Dict, Optional

from theseus.base.metrics.metric_template import Metric

class F1ScoreMetric(Metric):
    """
    F1 Score Metric (including macro, micro)
    """
    def __init__(self, average = 'weighted', **kwargs):
        super().__init__(**kwargs)
        self.average = average
        self.reset()

    def update(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        targets = batch["targets"] 
        outputs = outputs["outputs"] 

        outputs = torch.argmax(outputs,dim=1)
        outputs = outputs.detach().cpu()
        targets = targets.detach().cpu().view(-1)
    
        self.preds +=  outputs.numpy().tolist()
        self.targets +=  targets.numpy().tolist()

    def value(self):
        score = f1_score(self.targets, self.preds, average=self.average)
        return {f"{self.average}-f1": score}

    def reset(self):
        self.targets = []
        self.preds = []