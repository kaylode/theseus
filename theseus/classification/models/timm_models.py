from typing import Dict, List, Any, Optional
import timm
import torch
import torch.nn as nn
from theseus.utilities.hooks import postfix_hook
from theseus.utilities.cuda import move_to

class BaseTimmModel(nn.Module):
    """Convolution models from timm
    
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
        from_pretrained: bool = True,
        classnames: Optional[List] = None,
        **kwargs
    ):
        super().__init__()
        self.name = name

        self.classnames = classnames

        if num_classes != 1000:
            self.model = timm.create_model(name, pretrained=from_pretrained, num_classes=num_classes)
        else:
            self.model = timm.create_model(name, pretrained=from_pretrained)

        # Register a postfix hook to extract model features when forward
        self.model.forward_features = postfix_hook(
            self.model.forward_features, self.get_feature_hook)
        
        self.features = None
        self.pooling = torch.nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def get_feature_hook(self, parameter):
        """
        A hook function to extract features, only a workaround
        """
        self.features = self.pooling(parameter)
        return parameter

    def get_model(self):
        """
        Return the full architecture of the model, for visualization
        """
        return self.model

    def forward(self, x: torch.Tensor):
        self.features = None # Clear current features
        outputs = self.model(x)
        return {
            'outputs': outputs,
            'features': self.features
        }

    def get_prediction(self, adict: Dict[str, Any], device: torch.device):
        """
        Inference using the model.

        adict: `Dict[str, Any]`
            dictionary of inputs
        device: `torch.device`
            current device 
        """
        inputs = move_to(adict['inputs'], device)
        outputs = self.model(inputs)

        probs, outputs = torch.max(torch.softmax(outputs, dim=1), dim=1)

        probs = probs.cpu().detach().numpy()
        classids = outputs.cpu().detach().numpy()

        if self.classnames:
            classnames = [self.classnames[int(clsid)] for clsid in classids]
        else:
            classnames = []

        return {
            'labels': classids,
            'confidences': probs, 
            'names': classnames,
        }
