import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

class DiceScore():
    def __init__(self, num_classes, ignore_index = None, eps=1e-6, thresh = 0.5):
        self.thresh = thresh
        self.num_classes = num_classes
        
        
        self.ignore_index = ignore_index
        self.eps = eps

        self.scores_list = []
        self.reset()

    def compute(self, outputs, targets): 
        # outputs: (batch, num_classes, W, H)
        # targets: (batch, num_classes, W, H)
      
        batch_size, _ , w, h = outputs.shape
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)
      
        one_hot = torch.zeros(batch_size, self.num_classes+1, h, w)
        one_hot.scatter_(1, targets.long(), 1)
        one_hot = one_hot[:, 1:] # ignore zero index which is background

        
        for cl in range(self.num_classes):
            cl_output = outputs[:,cl,:,:]
            cl_target = one_hot[:,cl,:,:]
            score = self.binary_compute(cl_output, cl_target)
            self.scores_list.append(np.array([score]))
        

    def binary_compute(self, output, target):
        # outputs: (batch, 1, W, H)
        # targets: (batch, 1, W, H)

        predict = (output > self.thresh).float()
        intersect = (predict * target).sum()
        union = (predict + target).sum()
        return 2. * intersect / union
        
    def reset(self):
        self.scores_list = []
        self.sample_size = 0

    def update(self, outputs, targets):
        self.sample_size += outputs.shape[0]
        self.compute(outputs, targets)

    def value(self):
        scores_list = np.array(self.scores_list)
        values = sum(scores_list) / self.sample_size #mean

        return {"dice_score" : np.round(values, decimals=4)}

    def summary(self):
        scores_list = np.array(self.scores_list)
        class_iou = sum(scores_list) / self.sample_size

        print(f'{self.value()}')
        for i, x in enumerate(class_iou):
            print(f'\tClass {i:3d}: {x:.4f}')

    def __str__(self):
        return f'Dice Score: {self.value()}'

    def __len__(self):
        return len(self.sample_size)

    