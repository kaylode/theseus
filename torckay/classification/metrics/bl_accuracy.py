import torch
import numpy as np
from typing import Any, Dict, Optional
from torckay.base.metrics.metric_template import Metric

def compute_multiclass(outputs, targets, index):
    correct = 0
    sample_size = 0
    for i,j in zip(outputs, targets):
        if j == index:
            sample_size+= 1
            if i == j:
                correct+=1
    return correct, sample_size

class BalancedAccuracyMetric(Metric):
    """
    Balanced Accuracy metric for classification
    """
    def __init__(self, num_classes):
        
        self.num_classes = num_classes
        self.reset()

    def update(self, outputs: torch.Tensor, batch: Dict[str, Any]):
        targets = batch["target"] 
        outputs = torch.argmax(outputs,dim=1)
        outputs = outputs.detach().cpu()
        targets = targets.detach().cpu().view(-1)
    
        self.outputs +=  outputs.numpy().tolist()
        self.targets +=  targets.numpy().tolist()

    def reset(self):
        self.outputs = []
        self.targets = []
        self.corrects = [0 for i in range(self.num_classes)]
        self.total = [0 for i in range(self.num_classes)]

    def value(self):
        for i in range(self.num_classes):
            correct, sample_size = compute_multiclass(self.outputs, self.targets, i)
            self.corrects[i] += correct
            self.total[i] += sample_size
        each_acc = [self.corrects[i]*1.0/(self.total[i]) for i in range(self.num_classes) if self.total[i]>0]
        values = sum(each_acc)/self.num_classes

        return {'bl_acc': values}