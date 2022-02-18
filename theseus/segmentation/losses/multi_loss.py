from typing import Dict, Any, Iterable
import torch
import torch.nn as nn

class MultiLoss(nn.Module):
    """Wrapper class for combining multiple loss function 
    
    """
    def __init__(self, losses: Iterable[nn.Module], weights=None, **kwargs):
        super().__init__()
        self.losses = losses
        self.weights = [1.0 for _ in range(len(losses))] if weights is None else weights

    def forward(self, pred: torch.Tensor, batch: Dict[str, Any], device: torch.device):
        """
        Forward inputs and targets through multiple losses
        """
        total_loss = 0
        total_loss_dict = {}

        for weight, loss_fn in zip(self.weights, self.losses):
          loss, loss_dict = loss_fn(pred, batch, device)
          total_loss += (weight*loss)
          total_loss_dict.update(loss_dict)

        total_loss_dict.update({'Total': total_loss.item()})
        return total_loss, total_loss_dict