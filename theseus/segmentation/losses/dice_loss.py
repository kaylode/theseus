import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Source: https://github.com/sithu31296/semantic-segmentation/blob/958ed542aa68003eb0a2b0799cf5cecfe6c7587c/semseg/losses.py
    """
    def __init__(self, eps=1e-6, **kwargs):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, predict, batch, device):
        labels = batch["targets"].to(device)
        inputs = F.softmax(predict, dim=1)  

        tp = torch.sum(labels*inputs, dim=(2, 3))
        fn = torch.sum(labels*(1-inputs), dim=(2, 3))
        fp = torch.sum((1-labels)*inputs, dim=(2, 3))

        dice_score = (tp + 1e-6) / (tp +  fn + fp + 1e-6)
        dice_score = torch.sum(1 - dice_score, dim=-1)

        loss = dice_score / labels.shape[1]
        loss = loss.mean()
        
        loss_dict = {"DICE": loss.item()}
        return loss, loss_dict