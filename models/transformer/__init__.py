import copy
import torch.nn as nn
from .embedding import Embeddings, PositionalEncoding
from .layers import EncoderLayer, DecoderLayer
from .norm import LayerNorm

def get_clones(module, N):
    """
    "Produce N identical layers."
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    """
    Core encoder is a stack of N EncoderLayers
    :input:
        vocab_size:     size of source vocab
        d_model:        embeddings dim
        d_ff:           feed-forward dim
        N:              number of layers
        heads:          number of attetion heads
        dropout:        dropout rate
    :output:
        encoded embeddings shape [batch * input length * model_dim]
    """
    def __init__(self, vocab_size, d_model, d_ff, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embeddings(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout_rate=dropout)
        self.layers = get_clones(EncoderLayer(d_model, d_ff, heads, dropout), N)
        self.norm = LayerNorm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    """
    Decoder with N-stacked DecoderLayers
    :input:
        vocab_size:     size of target vocab
        d_model:        embeddings dim
        d_ff:           feed-forward dim
        N:              number of layers
        heads:          number of attetion heads
        dropout:        dropout rate
    :output:
        decoded embeddings shape [batch * input length * model_dim]
    """
    def __init__(self, vocab_size, d_model, d_ff, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embeddings(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout_rate=dropout)
        self.layers = get_clones(DecoderLayer(d_model, d_ff, heads, dropout), N)
        self.norm = LayerNorm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    """
    Transformer model
    :input:
        src_vocab:     size of source vocab
        trg_vocab:     size of target vocab
        d_model:       embeddings dim
        d_ff:          feed-forward dim
        N:             number of layers
        heads:         number of attetion heads
        dropout:       dropout rate
    :output:
        next words probability shape [batch * input length * vocab_dim]
    """
    def __init__(self, src_vocab, trg_vocab, d_model, d_ff, N, heads, dropout):
        super().__init__()
        self.name = "Transformer"
        self.encoder = Encoder(src_vocab, d_model, d_ff, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, d_ff, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
        self.init_params()

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)