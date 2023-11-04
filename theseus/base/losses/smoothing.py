from typing import Any, Dict, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from theseus.base.utilities.cuda import move_to

""" Cross Entropy w/ smoothing or soft targets
Hacked together by / Copyright 2021 Ross Wightman
"""


class LabelSmoothingCrossEntropy(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

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

        logprobs = F.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        loss = loss.mean()

        loss_dict = {"SmoothCE": loss.item()}
        return loss, loss_dict


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

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

        loss = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)
        loss_dict = {"SoftCE": loss.item()}
        return loss, loss_dict
