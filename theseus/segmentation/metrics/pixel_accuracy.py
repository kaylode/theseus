import torch
import numpy as np
from typing import Any, Dict, Optional

from theseus.base.metrics.metric_template import Metric

class PixelAccuracy(Metric):
    """Accuracy for each pixel comparision
    
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
        self.sample_size += outputs.shape[0]

        batch_size, _ , w, h = outputs.shape
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)
      
        one_hot_targets = torch.zeros(batch_size, self.num_classes, h, w)
        one_hot_predicts = torch.zeros(batch_size, self.num_classes, h, w)
        
        if self.pred_type == 'binary':
            predicts = (outputs > self.thresh).float()
        elif self.pred_type =='multi':
            predicts = torch.argmax(outputs, dim=1).unsqueeze(1)

        predicts = predicts.detach().cpu()
        one_hot_targets.scatter_(1, targets.long(), 1)
        one_hot_predicts.scatter_(1, predicts.long(), 1)
        
        for cl in range(self.num_classes):
            cl_pred = one_hot_predicts[:,cl,:,:]
            cl_target = one_hot_targets[:,cl,:,:]
            score = self.binary_compute(cl_pred, cl_target)
            self.scores_list[cl] += sum(score)

    def binary_compute(self, predict: torch.Tensor, target: torch.Tensor):
        # predict: (batch, 1, W, H)
        # targets: (batch, 1, W, H)

        correct = (predict == target).sum((-2,-1))
        total = target.shape[-1] * target.shape[-2]
        return (correct + self.eps) *1.0 / (total +self.eps)
        
    def reset(self):
        self.scores_list = np.zeros(self.num_classes)
        self.sample_size = 0

    def value(self):
        scores_each_class = self.scores_list / self.sample_size #mean over number of samples
        if self.pred_type == 'binary':
            scores = scores_each_class[1] # ignore background which is label 0
        else:
            if self.ignore_index is not None:
                scores_each_class[self.ignore_index] = 0
                scores = sum(scores_each_class) / (self.num_classes - 1)
            else:
                scores = sum(scores_each_class) / self.num_classes

        return {"pixel_acc" : scores}