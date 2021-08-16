import torch.nn as nn
import torch.nn.functional as F

import seaborn
import matplotlib.pyplot as plt

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

def draw_attention_map(input, target, model, show_fig=True, save_fig=None, return_figs=False):
    """
    Draw attention map from model, default: transformer
    :input:
        input (str): input raw text
        target (str): target raw text
    """

    # Tokenize
    sent = input.split()
    tgt_sent = target.split()

    def draw(data, x, y, ax):
        """
        Seaborn draw 
        """
        seaborn.heatmap(data, 
                        xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
                        cbar=False, ax=ax)
    
    if return_figs:
        figs = []

    for layer in range(1, 6, 2):
        # print("Decoder Src Layer", layer+1)
        fig, axs = plt.subplots(1,4, figsize=(20, 10))
        for h in range(4):
            draw(model.decoder.layers[layer].attn_2.attn[0, h].data.cpu()[:len(tgt_sent), :len(sent)], 
                sent, tgt_sent if h ==0 else [], ax=axs[h])
        if show_fig:
            plt.show()
        if return_figs:
            figs.append([f"Layer {layer+1}", fig])

    if save_fig is not None:
        plt.savefig(save_fig)
    
    if return_figs:
        return figs