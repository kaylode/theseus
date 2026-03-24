from typing import Any

import torch
from torch import nn

from theseus.base.utilities.cuda import move_to


class MeanSquaredErrorLoss(nn.Module):
    r"""MSELoss is warper of mean square error loss"""

    def __init__(self, **kwargs):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        device: torch.device = None,
    ):
        pred = outputs["outputs"]
        if device is not None:
            target = move_to(batch["targets"], device)
        else:
            target = batch["targets"]

        if pred.shape == target.shape:
            loss = self.criterion(pred, target)
        else:
            loss = self.criterion(pred.squeeze(-1), target.view(-1).contiguous())

        loss_dict = {"MSE": loss.item()}
        return loss, loss_dict
