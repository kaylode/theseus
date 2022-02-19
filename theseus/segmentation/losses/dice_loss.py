import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    """
    def __init__(self, eps=1e-6, **kwargs):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, predict, batch, device):
        targets = batch["targets"].to(device)
        
        inputs = F.softmax(predict, dim=1)       
        
        #flatten label and prediction tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.eps)/(inputs.sum() + targets.sum() + self.eps)  
        
        loss =  1 - dice

        loss_dict = {"DICE": loss.item()}
        return loss, loss_dict