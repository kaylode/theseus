from typing import Any, Dict
import torch
from torch import nn
from theseus.utilities.cuda import move_to

class ClassificationCELoss(nn.Module):
    r"""CELoss is warper of cross-entropy loss"""

    def __init__(self, **kwargs):
        super(ClassificationCELoss, self).__init__(**kwargs)
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

