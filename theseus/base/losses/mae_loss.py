from typing import Any, Dict

import torch
from torch import nn

from theseus.base.utilities.cuda import move_to


class MeanAbsoluteErrorLoss(nn.Module):
    r"""MSELoss is warper of mean absolute error loss"""

    def __init__(self, **kwargs):
        super(MeanAbsoluteErrorLoss, self).__init__()

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
            target = batch["targets"]

        if pred.shape == target.shape:
            loss = torch.mean(torch.abs(pred - target))
        else:
            # If the shapes are different, we can use MSELoss
            loss = torch.mean(torch.abs(pred.squeeze(-1) - target.view(-1).contiguous()))


        loss_dict = {"MAE": loss.item()}
        return loss, loss_dict
