from typing import Any, Dict
import torch
import torch.nn as nn
from torchvision.ops.focal_loss import sigmoid_focal_loss
from theseus.utilities.cuda import move_to

class FocalLoss(nn.Module):
    r"""FocalLoss"""
    def __init__(self, alpha= 0.25, gamma=2, reduction='mean', **kwargs):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, outputs: Dict[str, Any], batch: Dict[str, Any], device: torch.device):
        outputs = outputs['outputs']
        targets = move_to(batch['targets'], device)
        num_classes = outputs.shape[-1]

        # Need to be one hot encoding
        if outputs.shape != targets.shape:
            targets = nn.functional.one_hot(targets, num_classes=num_classes)
            targets = targets.float().squeeze()

        loss = sigmoid_focal_loss(outputs, targets, self.alpha, self.gamma, self.reduction)
        loss_dict = {"L": loss.item()}
        return loss, loss_dict