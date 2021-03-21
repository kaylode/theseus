import torch
import numpy as np

def compute_multiclass(outputs, targets, index):
    correct = 0
    sample_size = 0
    for i,j in zip(outputs, targets):
        if j == index:
            sample_size+= 1
            if i == j:
                correct+=1
    return correct, sample_size


class BalancedAccuracyMetric():
    """
    Balanced Accuracy metric for classification
    """
    def __init__(self, num_classes, decimals = 10):
        
        self.num_classes = num_classes
        self.decimals = decimals
        self.reset()
        
    def compute(self, outputs, targets):
        for i in range(self.num_classes):
            correct, sample_size = compute_multiclass(outputs, targets, i)
            self.corrects[i] += correct
            self.total[i] += sample_size

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
        self.corrects = [0 for i in range(self.num_classes)]
        self.total = [0 for i in range(self.num_classes)]

    def value(self):
        self.compute(self.outputs, self.targets)
        each_acc = [self.corrects[i]*1.0/(self.total[i]+0.0001) for i in range(self.num_classes)]
        values = sum(each_acc)/self.num_classes

        return {"bl_acc" : np.around(values, decimals = self.decimals)}

    def __str__(self):
        return f'BL_Accuracy: {self.value()}'

    def __len__(self):
        return len(self.sample_size)

if __name__ == '__main__':
    accuracy = BalancedAccuracyMetric(num_classes=8, decimals = 4)

    out = [[1,4,2],[5,7,4],[2,3,0]]
    label = [1, 0, 0]
    outputs = torch.LongTensor(out)
    targets = torch.LongTensor(label)
    accuracy.update(outputs, targets)
    print(accuracy.value())