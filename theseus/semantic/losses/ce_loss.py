from typing import Dict, List
import torch
from torch import nn
from theseus.utilities.cuda import move_to

class CELoss(nn.Module):
    r"""CELoss is warper of cross-entropy loss"""

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(CELoss, self).__init__()
        self.weight = weight
        if self.weight is not None:
            self.weight = torch.FloatTensor(self.weight)
        self.ignore_index = ignore_index

    def forward(self, outputs, batch, device):
        pred = outputs["outputs"]
        target = move_to(batch["targets"], device)

        if self.weight is not None:
            self.weight = move_to(self.weight, device)

        if self.ignore_index is not None:
            target = torch.argmax(target, dim=1)
            loss = nn.functional.cross_entropy(pred, target, weight=self.weight, ignore_index=self.ignore_index)
        else:
            loss = nn.functional.cross_entropy(pred, target, weight=self.weight)
            
        loss_dict = {"CE": loss.item()}
        return loss, loss_dict

class SmoothCELoss(nn.Module):
    r"""SmoothCELoss is warper of label smoothing cross-entropy loss"""

    def __init__(self, alpha = 1e-6, ignore_index = None, reduction = "mean", **kwargs):
        super(SmoothCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, outputs, batch, device):
        pred = outputs["outputs"]
        targets = move_to(batch["targets"], device)
        
        batch_size, num_classes = pred.shape[:2]
        y_hot = move_to(torch.zeros(pred.shape), device).scatter_(1, targets.unsqueeze(1) , 1.0)
        y_smooth = (1 - self.alpha) * y_hot + self.alpha / num_classes
        loss = torch.sum(- y_smooth * torch.nn.functional.log_softmax(pred, -1), -1).sum()

        if self.reduction == "mean":
            loss /= batch_size

        loss_dict = {"CE": loss.item()}
        return loss, loss_dict

class OhemCELoss(nn.Module):
    """
    Cross-entropy loss with Oline Hard Example Mining.

    ignore_label: `int` 
        ignore index
    weight: `List` 
        weight for each class
    thresh: `float`
        threshold to consider hard negative
    """
    def __init__(self, ignore_label: int = 255, weight: List = None, thresh: float = 0.7, **kwargs) -> None:
        super().__init__()

        self.weight = weight
        if self.weight is not None:
            self.weight = torch.FloatTensor(self.weight)

        self.ignore_label = ignore_label
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=ignore_label, reduction='none')

    def forward(self, outputs: Dict, batch: Dict, device: torch.device) -> torch.Tensor:
        pred = outputs["outputs"]
        labels = move_to(batch["targets"], device)

        if len(labels.shape) != 3:
            labels = torch.argmax(labels, dim=1) 
        
        if self.weight is not None:
            self.criterion.weight = move_to(self.criterion.weight, device)

        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(pred, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        
        loss = loss_hard.mean()
        
        loss_dict = {"OhemCE": loss.item()}
        return loss, loss_dict