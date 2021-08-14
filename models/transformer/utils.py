import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    """
    Just a simple 2-layer feed forward, input and output shape are equal
    """
    def __init__(self, model_dim, ff_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(model_dim, ff_dim)
        self.w_2 = nn.Linear(ff_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Apply RELU and dropout between two layers
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))