from typing import Any, Dict, Optional
import torch
import numpy as np
from theseus.utilities.cuda import move_to
from theseus.base.metrics.metric_template import Metric

class DiceScore(Metric):
    """ Dice score metric for segmentation
    """
    def __init__(self, 
            num_classes: int, 
            calc_each_class: bool = False,
            **kwawrgs):

        self.calc_each_class = calc_each_class
        self.num_classes = num_classes
        self.reset()

    def update(self, outputs: Dict[str, Any], batch: Dict[str, Any]): 
        """
        Perform calculation based on prediction and targets
        """

        targets = batch['targets']
        outputs = outputs['outputs']
        
        if len(outputs.shape) == 4: # prob
            preds = torch.argmax(outputs, dim=1)
            targets = torch.argmax(targets, dim=1)
            preds = move_to(preds, torch.device('cpu'))
        else: #argmaxed
            targets = targets.permute(3,0,1,2).long().squeeze()
            preds = torch.from_numpy(outputs).long()

        one_hot_predicts = torch.nn.functional.one_hot(
              preds.long(), 
              num_classes=self.num_classes).permute(0, 3, 1, 2)

        one_hot_targets = torch.nn.functional.one_hot(
              targets.long(), 
              num_classes=self.num_classes).permute(0, 3, 1, 2)

        for cl in range(1, self.num_classes):
            cl_pred = one_hot_predicts[:,cl,:,:]
            cl_target = one_hot_targets[:,cl,:,:]
            score = self.binary_compute(cl_pred, cl_target)
            self.scores_list[cl] += score

        self.sample_size += 1 # batch size equals 1 in our experiments
        

    def binary_compute(self, predict: torch.Tensor, target: torch.Tensor):
        # outputs: (batch, W, H)
        # targets: (batch, W, H)

        if torch.sum(predict)==0 and torch.sum(target)==0:
            return 1.0
        elif torch.sum(target)==0 and torch.sum(predict)>0:
            return 0.0
        else:
            volume_sum = target.sum() + predict.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (target & predict).sum()
            return 2*volume_intersect / volume_sum
        
    def reset(self):
        self.scores_list = np.zeros(self.num_classes) 
        self.sample_size = 0

    def value(self):
        scores_each_class = self.scores_list / self.sample_size #mean over number of samples
        scores = sum(scores_each_class) / (self.num_classes - 1) # subtract background

        if self.calc_each_class:
            result_dict = {}
            result_dict.update({
                f'dice_{i}': scores_each_class[i]  for i in range(1, self.num_classes)
            })
            result_dict.update({
                'dice-avg': scores
            })
            return result_dict
        else:
            return {"dice" : scores}