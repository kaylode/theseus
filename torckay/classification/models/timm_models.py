from collections import OrderedDict
import timm
import torch
import torch.nn as nn

class BaseTimmModel(nn.Module):
    """Some Information about BaseTimmModel"""

    def __init__(
        self,
        num_classes=1000,
        name="efficientnet_b0",
        from_pretrained=True,
        classnames=None,
        **kwargs
    ):
        super().__init__()
        self.name = name

        self.classnames = classnames

        if num_classes != 1000:
            self.model = timm.create_model(name, pretrained=from_pretrained, num_classes=num_classes)
        else:
            self.model = timm.create_model(name, pretrained=from_pretrained)

    def get_model(self):
        return self.model

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    def get_prediction(self, adict, device):
        inputs = adict['inputs'].to(device)
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
