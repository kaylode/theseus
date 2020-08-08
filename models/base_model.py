import torch
import torch.nn as nn
import torch.utils.data as data

from metrics.classification import AccuracyMetric



class BaseModel(nn.Module):
    def __init__(self,
                optimizer,
                criterion,
                metrics = AccuracyMetric(),
                lr = 1e-4,
                device = None,
                freeze = False):

        super(BaseModel, self).__init__()
        
        self.lr = lr
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.freeze = freeze
        self.metrics = metrics
        if not isinstance(metrics, list):
            self.metrics = [metrics,]
        
        if device:
            self.criterion.to(device)

    def unfreeze(self):
        for params in self.parameters():
            params.requires_grad = True

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def update_metrics(self, outputs, targets):
        metric_dict = {}
        for metric in self.metrics:
            metric.update(outputs, targets)
            metric_dict.update(metric.value())
        return metric_dict
    
    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()