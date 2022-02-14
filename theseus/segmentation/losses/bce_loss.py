from typing import Dict, Any
import torch
import torch.nn as nn
from .dice_loss import DiceLoss

class BCEwithDiceLoss(nn.Module):
    """Binary Cross-entropy loss merged with dice loss
    
    """
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCELoss()

    def forward(self, pred: torch.Tensor, batch: Dict[str, Any], device: torch.device):
        target = batch["targets"].to(device)
        dice_loss, dice_dict = self.dice(pred, target)
        bce_loss, bce_dict = self.bce(pred, target)
        total_loss = dice_loss + bce_loss
        loss_dict = {"L": total_loss.item()}
        loss_dict.update(dice_dict)
        loss_dict.update(bce_dict)
        return total_loss, loss_dict