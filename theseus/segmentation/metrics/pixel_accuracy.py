import torch
import numpy as np
from typing import Any, Dict, Optional

from theseus.base.metrics.metric_template import Metric

class PixelAccuracy(Metric):
    """Accuracy for each pixel comparision, return Average Precision and Recall
    
    num_classes: `int` 
        number of classes
    thresh: `Optional[float]`
        threhold for binary segmentation

    """
    def __init__(self, 
            num_classes: int, 
            thresh: Optional[float] = None, 
            eps: float = 1e-6,
            ignore_index: Optional[int] = None,
            **kwargs):

        self.thresh = thresh
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.pred_type = "multi" if num_classes > 1 else "binary"
        self.eps = eps

        if self.pred_type == 'binary':
            assert thresh is not None, "Threshold should be specified for binary segmentation"

        if num_classes == 1:
            self.num_classes+=1
        
        self.reset()

    def update(self, outputs: torch.Tensor, batch: Dict[str, Any]): 
        """
        Perform calculation based on prediction and targets
        """
        # outputs: (batch, num_classes, W, H)
        # targets: (batch, num_classes, W, H)

        targets = batch['targets']
        assert len(targets.shape) == 4, "Wrong shape for targets"
        assert len(outputs.shape) == 4, "Wrong shape for targets"
        self.sample_size += outputs.shape[0]
      
        if self.pred_type == 'binary':
            predicts = (outputs > self.thresh).float()
        elif self.pred_type =='multi':
            predicts = torch.argmax(outputs, dim=1) 

        predicts = predicts.detach().cpu()

        one_hot_predicts = torch.nn.functional.one_hot(
              predicts.long(), 
              num_classes=self.num_classes).permute(0, 3, 1, 2)
        
        for cl in range(self.num_classes):
            cl_pred = one_hot_predicts[:,cl,:,:]
            cl_target = targets[:,cl,:,:]
            p, r = self.binary_compute(cl_pred, cl_target)
            self.precisions[cl] += p
            self.recalls[cl] += r

    def binary_compute(self, predict: torch.Tensor, target: torch.Tensor):
        # predict: (batch, W, H)
        # targets: (batch, W, H)

        tp = torch.sum(target*predict, dim=(-1, -2))
        fn = torch.sum(target*(1-predict), dim=(-1, -2))
        fp = torch.sum((1-target)*predict, dim=(-1, -2))

        precision = tp * 1.0 / (tp + fp + self.eps) 
        recall = tp * 1.0 / (tp + fn + self.eps)

        return torch.sum(precision), torch.sum(recall) # sum over batch
        
    def reset(self):
        self.precisions = np.zeros(self.num_classes)
        self.recalls = np.zeros(self.num_classes)
        self.sample_size = 0

    def value(self):
        precision_each_class = self.precisions / self.sample_size #mean over number of samples
        recall_each_class = self.recalls / self.sample_size #mean over number of samples

        # Mean over classes
        if self.pred_type == 'binary':
            precision = precision_each_class[1] # ignore background which is label 0
            recall = recall_each_class[1] # ignore background which is label 0
        else:
            if self.ignore_index is not None:
                precision_each_class[self.ignore_index] = 0
                recall_each_class[self.ignore_index] = 0

                precision = sum(precision_each_class) / (self.num_classes - 1)
                recall = sum(recall_each_class) / (self.num_classes - 1)
            else:
                precision = sum(precision_each_class) / self.num_classes
                recall = sum(recall_each_class) / self.num_classes

        return {"precision" : precision, "recall": recall}