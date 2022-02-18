import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101


def create_model(name, **kwargs):
    if name == 'deeplabv3_resnet50':
        return deeplabv3_resnet50(**kwargs)
    elif name == 'deeplabv3_resnet101':
        return deeplabv3_resnet101(**kwargs)
    else:
        raise ValueError("Wrong model name")

class DeepLabV3(nn.Module):
    """Wrapper model for DeepLabv3
    
    name: `str`
        timm model name
    num_classes: `int`
        number of classes
    from_pretrained: `bool` 
        whether to use timm pretrained
    classnames: `Optional[List]`
        list of classnames
    """

    def __init__(
        self,
        name: str,
        num_classes: int = 1000,
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.model = create_model(name, num_classes=num_classes)

    def get_model(self):
        return self.model

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        return outputs['out']

    def get_prediction(self, adict: Dict[str, Any], device: torch.device):
        inputs = adict['inputs'].to(device)
        thresh = adict['thresh']
        outputs = self.model(inputs)
        outputs = outputs['out']

        if self.num_classes == 1:
            predicts = (outputs > thresh).float()
        else:
            predicts = torch.argmax(outputs, dim=1)

        predicts = predicts.detach().cpu().squeeze().numpy()
        return {
            'masks': predicts
        }