import torch
import numpy as np

class AccuracyMetric():
    def __init__(self, decimals = 10):
        self.reset()
        self.decimals = decimals

    def calculate(self, output, target):
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum()
        sample_size = output.size(0)
        return correct, sample_size

    def update(self,  output, target):
        assert isinstance(output, torch.Tensor), "Please input tensors"
        value = self.calculate(output, target)
        self.correct += value[0]
        self.sample_size += value[1]
    def reset(self):
        self.correct = 0.0
        self.sample_size = 0.0

    def value(self):
        values = self.correct / self.sample_size
        return np.around(values.numpy(), decimals = self.decimals)

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
    accuracy.update(outputs, targets)
    print(accuracy)
    
  
