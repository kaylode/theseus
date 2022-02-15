from typing import Any, Dict, Optional
import torch
from torch import nn
from timm.loss import LabelSmoothingCrossEntropy

class CELoss(nn.Module):
    r"""CELoss is warper of cross-entropy loss"""

    def __init__(self, **kwargs):
        super(CELoss, self).__init__(**kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, batch: Dict[str, Any], device: torch.device):
        target = batch["targets"].to(device)
        loss = self.criterion(pred, target.view(-1).contiguous())
        loss_dict = {"L": loss.item()}
        return loss, loss_dict

class SmoothCELoss(nn.Module):
    r"""SmoothCELoss is warper of label smoothing cross-entropy loss"""

    def __init__(self, smoothing: float=0.1, **kwargs):
        super(SmoothCELoss, self).__init__(**kwargs)
        self.criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)

    def forward(self, pred, batch, device):
        target = batch["targets"].to(device)
        loss = self.criterion(pred, target.view(-1).contiguous())
        loss_dict = {"L": loss.item()}
        return loss, loss_dict