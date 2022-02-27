from typing import Any, Dict, Optional
import torch
import numpy as np
from theseus.base.metrics.metric_template import Metric

class mIOU(Metric):
    """ Mean IOU metric for segmentation
    num_classes: `int`
        number of classes 
    eps: `float`
        epsilon to avoid zero division
    thresh: `float`
        threshold for binary segmentation
    """
    def __init__(self, 
            num_classes: int, 
            eps: float = 1e-6, 
            thresh: Optional[float] = None,
            ignore_index: Optional[int] = None,
            **kwawrgs):

        self.thresh = thresh
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.pred_type = "multi" if self.num_classes > 1 else "binary"

        if self.pred_type == 'binary':
            assert thresh is not None, "Threshold should be specified for binary segmentation"
        if self.num_classes == 1:
            self.num_classes+=1

        self.eps = eps

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
            score = self.binary_compute(cl_pred, cl_target)
            self.scores_list[cl] += sum(score)
        

    def binary_compute(self, predict: torch.Tensor, target: torch.Tensor):
        # outputs: (batch, W, H)
        # targets: (batch, W, H)

        intersect = torch.sum(target*predict, dim=(-1, -2))
        A = torch.sum(target, dim=(-1, -2))
        B = torch.sum(predict, dim=(-1, -2))
        union = A + B - intersect
        return intersect / (union + self.eps)
        
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
        return {"miou" : scores}