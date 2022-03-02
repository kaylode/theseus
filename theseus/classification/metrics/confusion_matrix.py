import torch
from typing import Any, Dict, Optional, List
from sklearn.metrics import confusion_matrix
from theseus.base.metrics.metric_template import Metric
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def make_cm_fig(cm, labels: Optional[List] = None):
    """
    Make confusion matrix figure
    labels: `Optional[List]`
        classnames for visualization
    """
    fig, ax = plt.subplots(1, figsize=(10,10))

    ax = sns.heatmap(cm, annot=False, 
            fmt='', cmap='Blues',ax =ax)

    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted')
    ax.set_ylabel('Actual ');

    ## Ticket labels - List must be in alphabetical order
    if not labels:
        labels = [str(i) for i in range(len(cm))]

    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    return fig


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

class ConfusionMatrix(Metric):
    """
    Confusion Matrix metric for classification
    """
    def __init__(self, classnames=None, **kwargs):
        super().__init__(**kwargs)
        self.classnames = classnames
        self.num_classes = [i for i in range(len(self.classnames))] if classnames is not None else None
        self.reset()

    def update(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        # in torchvision models, pred is a dict[key=out, value=Tensor]
        outputs = outputs["outputs"] 
        targets = batch["targets"] 

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
        values = confusion_matrix(self.outputs, self.targets, labels=self.num_classes)
        fig = make_cm_fig(values, self.classnames)
        return {"cfm": fig}