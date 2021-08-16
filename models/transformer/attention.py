import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def attention(q, k, v, d_k, mask=None, dropout=None):
    """
    Calculate attention
    :input:
        q:          query
        k:          key
        v:          value
        d_k:        scaled term
        mask:       whether to use masking attention
        dropout:    dropout rate
    :output:
    """

    # Query, Key matrix multiplication
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    # If mask, use masking attetion
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e4)
    
    # Softmax for scaling in range [0,1]
    scores = F.softmax(scores, dim=-1)
    
    # Dropout
    if dropout is not None:
        scores = dropout(scores)

    # Score, Value matrix multiplication
    output = torch.matmul(scores, v)
    return output, scores

class MultiHeadAttention(nn.Module):
    """
    Calculate multihead attention with num_heads
    :input:
        heads:          number of attention heads
        d_model:        embedding dim
        dropout:        dropout rate
    :output:
    """
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        # For visualization
        self.attn = None
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention 
        scores, self.attn = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output