import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    def __init__(self, smooth=1, alpha=0.7, gamma=0.75, **kwargs):
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = 1-alpha
        self.gamma = gamma

    def forward(self, predict, batch, device):
        targets = batch["targets"].to(device)
        prediction = F.softmax(predict, dim=1)  
        
        #flatten label and prediction tensors
        prediction = prediction.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (prediction * targets).sum()    
        FP = ((1-targets) * prediction).sum()
        FN = (targets * (1-prediction)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha*FN + self.beta*FP + self.smooth)  
        loss = (1 - tversky)**self.gamma
        
        loss_dict = {"FT": loss.item()}
        return loss, loss_dict