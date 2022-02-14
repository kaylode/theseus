from typing import Dict, Any
import torch
import torch.nn as nn
from .dice_loss import DiceLoss, BinaryDiceLoss
from .ce_loss import CELoss

class BCEwithDiceLoss(nn.Module):
    """Binary Cross-entropy loss merged with dice loss
    
    """
    def __init__(self):
        super().__init__()
        self.dice = BinaryDiceLoss()
        self.ce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, batch: Dict[str, Any], device: torch.device):
        targets = batch['targets'].to(device)
        dice_loss, _ = self.dice(pred, batch, device)
        ce_loss = self.ce(pred, targets)
        total_loss = dice_loss + ce_loss
        loss_dict = {
          "DICE": dice_loss.item(),
          "CE": ce_loss.item(),
          "L": total_loss.item()}
        return total_loss, loss_dict