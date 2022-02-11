from typing import Any, Dict, Optional

import torch
from torckay.base.metrics.metric_template import Metric


class Accuracy(Metric):

    """Pixel accuracy metric

    Segmentation multi classes metric

    Args:
        nclasses (int): number of classé
        ignore_index (Optional[Any], optional): [description]. Defaults to None.
    """

    def __init__(self, ignore_index: Optional[Any] = None, **kwargs):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self.reset()

    def update(self, output: torch.Tensor, batch: Dict[str, Any]):

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