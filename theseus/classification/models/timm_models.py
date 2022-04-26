from typing import Dict, List, Any, Optional
import timm
import torch
import torch.nn as nn
from theseus.utilities.hooks import postfix_hook
from theseus.utilities.cuda import move_to, detach
from theseus.classification.utilities.logits import logits2labels
import torch.nn.functional as F

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
        model_name: str,
        num_classes: int = 1000,
        from_pretrained: bool = True,
        classnames: Optional[List] = None,
        freeze: bool = False,
        **kwargs
    ):
        super().__init__()
        self.name = model_name

        self.classnames = classnames
        self.num_classes = num_classes
        self.freeze = freeze

        if self.num_classes != 1000:
            self.model = timm.create_model(model_name, pretrained=from_pretrained, num_classes=self.num_classes)
        else:
            self.model = timm.create_model(model_name, pretrained=from_pretrained)

        self.feature_dim = self.model.num_features

        if self.num_classes > 0:
            # Register a postfix hook to extract model features when forward
            self.model.forward_features = postfix_hook(
                self.model.forward_features, self.get_feature_hook)
        
            self.features = None
            self.pooling = torch.nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )

        if self.freeze:
            self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = False

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

    def forward(self, batch: Dict, device: torch.device):
        x = move_to(batch['inputs'], device)
        self.features = None # Clear current features
        outputs = self.model(x)
        if self.num_classes == 0:
            self.features = outputs
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
        outputs = self.forward(adict, device)['outputs']

        if not adict['multilabel']:
            outputs, probs = logits2labels(outputs, label_type='multiclass', return_probs=True)
        else:
            outputs, probs = logits2labels(outputs, label_type='multilabel', threshold=adict['threshold'], return_probs=True)

            if adict['no-zeroes']:
                argmaxs = torch.argmax(probs, dim=1)
                tmp = torch.sum(outputs, dim=1)
                one_hots = F.one_hot(argmaxs, outputs.shape[1])
                outputs[tmp==0] = one_hots[tmp==0].bool()

        probs = move_to(detach(probs), torch.device('cpu')).numpy()
        classids = move_to(detach(outputs), torch.device('cpu')).numpy()

        if self.classnames and not adict['multilabel']:
            classnames = [self.classnames[int(clsid)] for clsid in classids]
        elif self.classnames and adict['multilabel']:
            classnames = [
              [self.classnames[int(i)] for i, c in enumerate(clsid) if c]
              for clsid in classids
            ]
        else:
            classnames = []

        return {
            'labels': classids,
            'confidences': probs, 
            'names': classnames,
        }
