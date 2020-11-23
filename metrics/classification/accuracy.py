import torch
import numpy as np

def compute_multiclass(output, target):
    pred = torch.argmax(output, dim=1)
    correct = (pred == target).sum()
    sample_size = output.size(0)
    return correct, sample_size

def compute_binary(output, target, thresh=0.7):
    
    pred = output >= thresh
    correct = (pred == target).sum()
    sample_size = output.size(0)
    return correct, sample_size

def get_compute(types):
    if types == 'binary':
        return compute_binary
    elif types == 'multi':
        return compute_multiclass

class AccuracyMetric():
    """
    Accuracy metric for classification
    """
    def __init__(self, types = 'multi', decimals = 10):
        self.reset()
        self.decimals = decimals
        self.compute_fn = get_compute(types)
        
    def compute(self, outputs, targets):
        return self.compute_fn(outputs.squeeze(1), targets)

    def update(self,  outputs, targets):
        assert isinstance(outputs, torch.Tensor), "Please input tensors"
        value = self.compute(outputs, targets)
        self.correct += value[0]
        self.sample_size += value[1]

    def reset(self):
        self.correct = 0.0
        self.sample_size = 0.0

    def value(self):
        values = self.correct / self.sample_size

        if values.is_cuda:
            values = values.cpu()
        return {"acc" : np.around(values.numpy(), decimals = self.decimals)}

    def __str__(self):
        return f'Accuracy: {self.value()}'

    def __len__(self):
        return len(self.sample_size)

if __name__ == '__main__':
    accuracy = AccuracyMetric(decimals = 4)
    out = [[1,4,2],[5,7,4],[2,3,0]]
    label = [[1, 1, 0]]
    outputs = torch.LongTensor(out)
    targets = torch.LongTensor(label)
    accuracy.update(outputs, targets)
    di = {}
    di.update(accuracy.value())
    print(di)
    
  
