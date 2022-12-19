from typing import Any, Dict
import numpy as np
from theseus.base.metrics.metric_template import Metric
from scipy.special import softmax

class SKLAccuracy(Metric):
    """
    Accuracy metric
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def value(self, output: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        output = output["outputs"] 
        target = batch["targets"] 
        probs = softmax(output, axis=-1)
        predictions = np.argmax(probs, axis=-1)
        correct = (predictions.reshape(-1) == target.reshape(-1)).sum()
        score = correct * 1.0 / target.shape[0]
        return {'acc': score}


def compute_multiclass(outputs, targets, index):
    correct = 0
    sample_size = 0
    for i,j in zip(outputs, targets):
        if j == index:
            sample_size+= 1
            if i == j:
                correct+=1
    return correct, sample_size

class SKLBalancedAccuracyMetric(Metric):
    """
    Balanced Accuracy metric for classification
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def value(self, outputs: Dict[str, Any], batch: Dict[str, Any]):

        outputs = outputs["outputs"] 
        targets = batch["targets"] 
        predictions = np.argmax(outputs,axis=-1).reshape(-1).tolist()
        targets = targets.reshape(-1).tolist()

        unique_ids = np.unique(targets)
        corrects = {str(k):0 for k in unique_ids}
        total = {str(k):0 for k in unique_ids}

        # Calculate accuracy for each class index
        for i in unique_ids:
            correct, sample_size = compute_multiclass(predictions, targets, i)
            corrects[str(i)] += correct
            total[str(i)] += sample_size
        each_acc = [corrects[str(i)]*1.0/(total[str(i)]) for i in unique_ids if total[str(i)]>0]

        # Get mean accuracy across classes
        values = sum(each_acc)/len(unique_ids)

        return {'bl_acc': values}