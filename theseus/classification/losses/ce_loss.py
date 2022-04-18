from typing import Any, Dict
import torch
from torch import nn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from theseus.utilities.cuda import move_to

class CELoss(nn.Module):
    r"""CELoss is warper of cross-entropy loss"""

    def __init__(self, **kwargs):
        super(CELoss, self).__init__(**kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs: Dict[str, Any], batch: Dict[str, Any], device: torch.device):
        pred = outputs["outputs"]
        target = move_to(batch["targets"], device)

        if pred.shape == target.shape:
            loss = self.criterion(pred, target)
        else:
            loss = self.criterion(pred, target.view(-1).contiguous())

        loss_dict = {"CE": loss.item()}
        return loss, loss_dict

class SmoothCELoss(nn.Module):
    r"""SmoothCELoss is warper of label smoothing cross-entropy loss"""

    def __init__(self, smoothing: float=0.1, **kwargs):
        super(SmoothCELoss, self).__init__(**kwargs)
        self.smooth_criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
        self.soft_criterion = SoftTargetCrossEntropy()

    def forward(self, outputs: Dict[str, Any], batch: Dict[str, Any], device: torch.device):
        pred = outputs['outputs']
        target = move_to(batch["targets"], device)

        if pred.shape == target.shape:
            loss = self.soft_criterion(pred, target)
        else:
            loss = self.smooth_criterion(pred, target.view(-1).contiguous())
        loss_dict = {"CE": loss.item()}
        return loss, loss_dict