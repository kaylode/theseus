from typing import Any, Dict, Optional

import torch
from theseus.base.metrics.metric_template import Metric


class Accuracy(Metric):
    """Accuracy metric
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset()

    def update(self, output: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        output = output["outputs"] 
        target = batch["targets"] 
        prediction = torch.argmax(output, dim=1)
        prediction = prediction.cpu().detach()

        correct = (prediction.view(-1) == target.view(-1)).sum()
        correct = correct.cpu()
        self.total_correct += correct
        self.sample_size += prediction.size(0)

    def value(self):
        return {'acc': (self.total_correct / self.sample_size).item()}

    def reset(self):
        self.total_correct = 0
        self.sample_size = 0