import torch
import numpy as np
from typing import Any, Dict, Optional
from sklearn.metrics import confusion_matrix
from torckay.base.metrics import METRIC_REGISTRY
from torckay.base.metrics.metric_template import Metric

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    result = ""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "
    
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    result += ("    " + fst_empty_cell + " ")
    # End CHANGES
    
    for label in labels:
        result += (("%{0}s".format(columnwidth) % label) + " ")
        
    result += '\n'
    # Print rows
    for i, label1 in enumerate(labels):
        result += (("    %{0}s".format(columnwidth) % label1) + " ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            result += (cell + " ")
        result += '\n'
    return result

@METRIC_REGISTRY.register()
class ConfusionMatrix(Metric):
    """
    Confusion Matrix metric for classification
    """
    def __init__(self, classes_map):
        self.classes_map = classes_map
        self.labels = [classes_map[i] for i in range(len(classes_map))]
        self.reset()

    def update(self, output: torch.Tensor, batch: Dict[str, Any]):

        outputs = output["out"] if isinstance(output, Dict) else output
        # in torchvision models, pred is a dict[key=out, value=Tensor]
        targets = batch["target"] if isinstance(batch, Dict) else batch

        outputs = torch.argmax(outputs,dim=1)
        if outputs.is_cuda:
            outputs = outputs.cpu()
            targets = targets.cpu()

        self.outputs +=  outputs.numpy().tolist()
        self.targets +=  targets.numpy().tolist()
        
    def reset(self):
        self.outputs = []
        self.targets = []

    def value(self):
        values = confusion_matrix(self.outputs, self.targets)
        return {"cfm": str(values)}