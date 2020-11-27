import torch
import numpy as np
from sklearn.metrics import confusion_matrix

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

class ConfusionMatrix():
    """
    Confusion Matrix metric for classification
    """
    def __init__(self, classes_map):
        self.classes_map = classes_map
        self.labels = [classes_map[i] for i in range(len(classes_map))]
        self.reset()
        
    def compute(self, outputs, targets):
        return confusion_matrix(outputs, targets)

    def update(self,  outputs, targets):
        assert isinstance(outputs, torch.Tensor), "Please input tensors"
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
        values = self.compute(self.targets, self.outputs)
        print("Confusion Matrix:")
        print(print_cm(values, self.labels))
        return {}

    def __str__(self):
        return print_cm(self.value()["confusion_matrix"], self.labels)

    def __len__(self):
        return len(self.sample_size)

if __name__ == '__main__':
    classes_map = {1:'cat',0:'dog',2:'human'}
    cm = ConfusionMatrix(classes_map)
    out = [[4,1,2],[5,2,4],[2,3,4],[0,1,4],[5,1,4],[3,3,5]]
    label = [2, 0, 2, 2, 0, 1]
    outputs = torch.LongTensor(out)
    targets = torch.LongTensor(label)
    cm.update(outputs, targets)
    
    print(cm.value())
    
   
    
  
