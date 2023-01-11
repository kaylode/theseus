from typing import Any, Dict

import torch
from torch import nn

from theseus.base.utilities.cuda import move_to

from .smoothing import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


class ClassificationCELoss(nn.Module):
    r"""CELoss is warper of cross-entropy loss"""

    def __init__(self, **kwargs):
        super(ClassificationCELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        device: torch.device,
    ):
        pred = outputs["outputs"]
        target = move_to(batch["targets"], device)

        if pred.shape == target.shape:
            loss = self.criterion(pred, target)
        else:
            loss = self.criterion(pred, target.view(-1).contiguous())

        loss_dict = {"CE": loss.item()}
        return loss, loss_dict


class ClassificationSmoothCELoss(nn.Module):
    r"""SmoothCELoss is warper of label smoothing cross-entropy loss"""

    def __init__(self, smoothing: float = 0.1, **kwargs):
        super(ClassificationSmoothCELoss, self).__init__()
        self.smooth_criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
        self.soft_criterion = SoftTargetCrossEntropy()

    def forward(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        device: torch.device,
    ):
        pred = outputs["outputs"]
        target = batch["targets"]

        if pred.shape == target.shape:
            loss, loss_dict = self.soft_criterion(
                {"outputs": pred}, {"targets": target}, device
            )
        else:
            # batch["targets"] = batch["targets"].view(-1).contiguous()
            loss, loss_dict = self.smooth_criterion(
                {"outputs": pred},
                {"targets": batch["targets"].view(-1).contiguous()},
                device,
            )
        return loss, loss_dict
