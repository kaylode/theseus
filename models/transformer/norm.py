import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, model_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(model_dim))
        self.b_2 = nn.Parameter(torch.zeros(model_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2