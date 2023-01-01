import torch
from torch import nn
from theseus.base.models.wrapper import ModelWithLoss

class ModelWithLossandPostprocess(ModelWithLoss):
    """Add utilitarian functions for module to work with pipeline
    Args:
        model (Module): Base Model without loss
        loss (Module): Base loss function with stat
    """

    def __init__(self, model: nn.Module, criterion: nn.Module, device: torch.device):
        super().__init__(model, criterion, device)

    def forward(self, batch, metrics=None):
        """
        Forward the batch through models, losses and metrics
        If some parameters are needed, it's best to include in the batch
        """
        outputs = self.model(batch, self.device)
        loss, loss_dict = self.criterion(outputs, batch, self.device)

        if metrics is not None:
            outputs, batch = self.model.postprocess(outputs=outputs, batch=batch)
            for metric in metrics:
                metric.update(output=outputs, batch=batch)

        return {
            'loss': loss,
            'loss_dict': loss_dict,
            'model_outputs': outputs
        }