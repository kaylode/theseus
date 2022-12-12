from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from theseus.utilities.cuda import move_to, detach
import torch.nn.functional as F

class NeuralNet(torch.nn.Module):
    def __init__(self, input_dim, num_classes, classnames=None, num_hidden_layers=0, hidden_dim=None):
        super(NeuralNet, self).__init__()

        self.num_classes=num_classes
        self.classnames = classnames
        
        if num_hidden_layers > 0:
            assert hidden_dim is not None, 'hidden_dim should be specified'
        if hidden_dim is not None:
            assert num_hidden_layers > 0, 'number of hidden layers should be specified'

        hidden_layers = None
        if num_hidden_layers > 0:
            hidden_layers = nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim) for i in range(num_hidden_layers)])

        if hidden_layers is not None:
            self.model = nn.Sequential(*[
                torch.nn.Linear(input_dim, hidden_dim),
                hidden_layers,
                torch.nn.Linear(hidden_dim, num_classes),
            ])
        else:
            self.model = torch.nn.Linear(input_dim, num_classes)

    def freeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def get_model(self):
        """
        Return the full architecture of the model, for visualization
        """
        return self.model

    def forward(self, batch: Dict, device: torch.device):
        x = move_to(batch['inputs'], device)
        outputs = self.model(x)
        return {
            'outputs': outputs,
        }
