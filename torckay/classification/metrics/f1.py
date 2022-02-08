import torch
import numpy as np
from sklearn.metrics import f1_score
from typing import Any, Dict, Optional

from torckay.base.metrics.metric_template import Metric

class F1ScoreMetric():
    """
    F1 Score Metric (including macro, micro)
    """
    def __init__(self, average = 'weighted'):
        self.average = average
        self.reset()

    def update(self, output: torch.Tensor, batch: Dict[str, Any]):

        outputs = output["out"] if isinstance(output, Dict) else output
        # in torchvision models, pred is a dict[key=out, value=Tensor]
        targets = batch["target"] if isinstance(batch, Dict) else batch

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