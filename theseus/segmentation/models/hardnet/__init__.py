from typing import List, Dict, Any
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .layer import ConvModule, HarDBlock

"""
Source: https://github.com/sithu31296/semantic-segmentation/blob/main/semseg/models/fchardnet.py
"""

class FCHarDNet(nn.Module):
    def __init__(self, num_classes: int = 19, **kwargs) -> None:
        super().__init__()
        first_ch, ch_list, gr, n_layers = [16, 24, 32, 48], [64, 96, 160, 224, 320], [10, 16, 18, 24, 32], [4, 4, 8, 8, 8]
        self.num_classes = num_classes
        self.base = nn.ModuleList([])

        # stem
        self.base.append(ConvModule(3, first_ch[0], 3, 2))
        self.base.append(ConvModule(first_ch[0], first_ch[1], 3))
        self.base.append(ConvModule(first_ch[1], first_ch[2], 3, 2))
        self.base.append(ConvModule(first_ch[2], first_ch[3], 3))

        self.shortcut_layers = []
        skip_connection_channel_counts = []
        ch = first_ch[-1]

        for i in range(len(n_layers)):
            blk = HarDBlock(ch, gr[i], n_layers[i])
            ch = blk.out_channels

            skip_connection_channel_counts.append(ch)
            self.base.append(blk)

            if i < len(n_layers) - 1:
                self.shortcut_layers.append(len(self.base) - 1)

            self.base.append(ConvModule(ch, ch_list[i], k=1))
            ch = ch_list[i]
            
            if i < len(n_layers) - 1:
                self.base.append(nn.AvgPool2d(2, 2))

        prev_block_channels = ch
        self.n_blocks = len(n_layers) - 1

        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up = nn.ModuleList([])

        for i in range(self.n_blocks-1, -1, -1):
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            blk = HarDBlock(cur_channels_count // 2, gr[i], n_layers[i])
            prev_block_channels = blk.out_channels
            
            self.conv1x1_up.append(ConvModule(cur_channels_count, cur_channels_count//2, 1))
            self.denseBlocksUp.append(blk)

        self.finalConv = nn.Conv2d(prev_block_channels, num_classes, 1, 1, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        H, W = x.shape[-2:]
        skip_connections = []
        for i, layer in enumerate(self.base):
            x = layer(x)
            if i in self.shortcut_layers:
                skip_connections.append(x)

        out = x

        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out = F.interpolate(out, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            out = torch.cat([out, skip], dim=1)
            out = self.conv1x1_up[i](out)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        return out

    def get_model(self):
        return self

    def get_prediction(self, adict: Dict[str, Any], device: torch.device):
        inputs = adict['inputs'].to(device)
        outputs = self.forward(inputs)

        if self.num_classes == 1:
            thresh = adict['thresh']
            predicts = (outputs > thresh).float()
        else:
            predicts = torch.argmax(outputs, dim=1)

        predicts = predicts.detach().cpu().squeeze().numpy()
        return {
            'masks': predicts
        } 