from typing import Any, Dict
import torch
from torch import nn
from theseus.base.utilities.cuda import move_to

class MeanSquaredErrorLoss(nn.Module):
    r"""MSELoss is warper of mean square error loss"""

    def __init__(self, **kwargs):
        super(MeanSquaredErrorLoss, self).__init__(**kwargs)
        self.criterion = nn.MSELoss() 

    def forward(self, outputs: Dict[str, Any], batch: Dict[str, Any], device: torch.device):
        pred = outputs["outputs"]
        target = move_to(batch["targets"], device)

        if pred.shape == target.shape:
            loss = self.criterion(pred, target)
        else:
            loss = self.criterion(pred, target.view(-1).contiguous())

        loss_dict = {"MSE": loss.item()}
        return loss, loss_dict