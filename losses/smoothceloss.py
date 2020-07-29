import torch.nn as nn
import torch


class smoothCELoss(nn.Module):
    """
    References: https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06
    """
    def __init__(self, alpha = 1e-6, ignore_index = None, reduction = "mean"):
        super(smoothCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, outputs, targets):
        # Outputs size: batch_size * num_classes
        # Targets size: batch_size

        batch_size, num_classes = outputs.shape
        y_hot = torch.zeros(outputs.shape).scatter_(1, targets.unsqueeze(1) , 1.0)
        y_smooth = (1 - alpha) * y_hot + alpha / num_classes
        loss = torch.sum(- y_smooth * torch.nn.functional.log_softmax(outputs, -1), -1).sum()

        if self.reduction:
            loss /= batch_size
  
        return loss
        