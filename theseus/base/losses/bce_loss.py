from typing import Any, Dict

import torch
from torch import nn

from theseus.base.utilities.cuda import move_to


class BCELoss(nn.Module):
    r"""CELoss is warper of cross-entropy loss"""

    def __init__(self, **kwargs):
        super(BCELoss, self).__init__()
        if "weight" in kwargs:
            weight = torch.FloatTensor(kwargs.get("weight"))
        else:
            weight = None
        self.criterion = nn.BCELoss(
            weight=weight,
        )

    def forward(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        device: torch.device = None,
    ):
        pred = outputs["outputs"]
        if device is not None:
            target = move_to(batch["targets"], device)
        else:
            target = batch["targets"].float()

        if pred.shape == target.shape:
            loss = self.criterion(pred, target)
        else:
            loss = self.criterion(pred, target.view(-1).contiguous())

        loss_dict = {"BCE": loss.item()}
        return loss, loss_dict
