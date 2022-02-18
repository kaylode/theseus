from typing import Dict
import torch
from torch import nn

class CELoss(nn.Module):
    r"""CELoss is warper of cross-entropy loss"""

    def __init__(self, **kwargs):
        super(CELoss, self).__init__()

    def forward(self, pred, batch, device):
        target = batch["targets"].to(device)

        loss = nn.functional.cross_entropy(pred, target)
        loss_dict = {"CE": loss.item()}
        return loss, loss_dict

class SmoothCELoss(nn.Module):
    r"""SmoothCELoss is warper of label smoothing cross-entropy loss"""

    def __init__(self, alpha = 1e-6, ignore_index = None, reduction = "mean", **kwargs):
        super(SmoothCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, pred, batch, device):
        targets = batch["targets"].to(device)
        
        batch_size, num_classes = pred.shape[:2]
        y_hot = torch.zeros(pred.shape).to(device).scatter_(1, targets.unsqueeze(1) , 1.0)
        y_smooth = (1 - self.alpha) * y_hot + self.alpha / num_classes
        loss = torch.sum(- y_smooth * torch.nn.functional.log_softmax(pred, -1), -1).sum()

        if self.reduction == "mean":
            loss /= batch_size

        loss_dict = {"CE": loss.item()}
        return loss, loss_dict