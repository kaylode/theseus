import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Source: https://github.com/sithu31296/semantic-segmentation/blob/958ed542aa68003eb0a2b0799cf5cecfe6c7587c/semseg/losses.py
    
    Note: If Loss becomes NaN, try changing other backbone/model or use different losses
    DiceLoss won't work everytime due to some activation/normalization layers in the architecture
    """
    def __init__(self, eps=1e-6, **kwargs):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, predict, batch, device):
        targets = batch["targets"].to(device)
        prediction = F.softmax(predict, dim=1)  

        # have to use contiguous since they may from a torch.view op
        iflat = prediction.contiguous().view(-1)
        tflat = targets.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        union = prediction.sum() + targets.sum()

        dice = (2.*intersection + self.eps)/(union + self.eps)  
        loss =  1 - dice

        loss_dict = {"DICE": loss.item()}
        return loss, loss_dict