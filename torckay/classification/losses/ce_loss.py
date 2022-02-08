from typing import Dict
from torch import nn
from torckay.classification.losses import LOSS_REGISTRY
from timm.loss import LabelSmoothingCrossEntropy

@LOSS_REGISTRY.register()
class CELoss(nn.Module):
    r"""CELoss is warper of cross-entropy loss"""

    def __init__(self):
        super(CELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, batch):
        target = batch["target"]
        loss = self.criterion(pred, target.view(-1))
        loss_dict = {"L": loss.item()}
        return loss, loss_dict

@LOSS_REGISTRY.register()
class SmoothCELoss(nn.Module):
    r"""SmoothCELoss is warper of label smoothing cross-entropy loss"""

    def __init__(self, smoothing=0.1):
        super(SmoothCELoss, self).__init__()
        self.criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)

    def forward(self, pred, batch):
        target = batch["target"]
        loss = self.criterion(pred, target.view(-1))
        loss_dict = {"L": loss.item()}
        return loss, loss_dict