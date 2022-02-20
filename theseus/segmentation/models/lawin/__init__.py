from typing import Dict, Any
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .head import LawinHead
from ..segformer.backbone import MiT
from ..segformer.utils import trunc_normal_

"""
https://github.com/sithu31296/semantic-segmentation/blob/main/semseg/models/lawin.py
"""

class Lawin(nn.Module):
    """
    Notes::::: This implementation has larger params and FLOPs than the results reported in the paper.
    Will update the code and weights if the original author releases the full code.
    """
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19) -> None:
        super().__init__()
        self.num_classes = num_classes
        backbone_name, variant = backbone.split('-')
        self.backbone = MiT(variant)
        self.decode_head = LawinHead(self.backbone.channels, 256 if 'B0' in backbone else 512, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

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