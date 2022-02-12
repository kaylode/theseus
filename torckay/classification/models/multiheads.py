from collections import OrderedDict
import timm
from timm.models.layers import SelectAdaptivePool2d

import torch
import torch.nn as nn
from torckay.utilities.loading import load_state_dict

from torckay.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger('main')


class MultiHeads(nn.Module):
    def __init__(self, backbone, num_head_classes, forward_index) -> None:
        super().__init__()
        self.num_head_classes = num_head_classes
        self.forward_index = forward_index

        # Create multiheads
        self.heads = nn.ModuleList()
        for i, num_classes in enumerate(num_head_classes):

            self.heads.add_module(f"{i}", self.create_head(backbone, num_classes))
            if forward_index != i:
                self.heads[i].requires_grad = False

    def create_head(self, model, num_classes):
        return nn.Sequential(OrderedDict([
                ('global_pool', SelectAdaptivePool2d(pool_type='avg')),
                ('norm', model.head.norm),
                ('flatten', nn.Flatten(1)),
                ('drop', nn.Dropout(model.drop_rate)),
                ('fc', nn.Linear(model.num_features, num_classes) if num_classes > 0 else nn.Identity())
            ]))
        

    def forward(self, x):
        return self.forward_head(x, self.forward_index)

    def forward_head(self, x, head_index):
        return self.heads[head_index](x)


class MultiHeadModel(nn.Module):
    """Some Information about BaseTimmModel"""

    def __init__(
        self,
        name="efficientnet_b0",
        pretrained_backbone=None,
        num_head_classes=[1,1],
        train_index=0,
        txt_classnames=None,
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.train_index = train_index
        self.txt_classnames = txt_classnames
        if txt_classnames is not None:
            self.load_classnames()

        model = timm.create_model(name, pretrained=True)
        self.drop_rate = model.drop_rate
        self.num_features = model.num_features

        # Remove last head, freeze backbone
        self.model = nn.Sequential()
        for n,m in list(model.named_children())[:-1]:
            self.model.add_module(n, m)

        for param in self.model.parameters():
            param.requires_grad = False

        if pretrained_backbone is not None:
            state_dict = torch.load(pretrained_backbone)
            load_state_dict(self, state_dict, 'model')

        self.feature_layer_name = list(self.model.named_children())[-1][0]

        heads = MultiHeads(model, num_head_classes, train_index)
        # Add heads to model
        self.model.add_module('heads', heads)

    def get_model(self):
        return self.model

    def load_classnames(self):
        self.classnames = []
        with open(self.txt_classnames, 'r') as f:
            groups = f.read().splitlines()

        for group in groups:
            classnames = group.split()
            self.classnames.append(classnames)

    def forward_features(self, x):
        features = None

        def forward_features_hook(module_, input_, output_):
            nonlocal features
            features = output_

        a_hook = self.model._modules[self.feature_layer_name].register_forward_hook(forward_features_hook) 
        
        self.model(x)

        a_hook.remove()
        return features

    def forward_head(self, x, head_index):
        
        features = self.forward_features(x)
        outputs = self.model.heads.forward_head(features, head_index)
        return outputs

    def forward(self, x):
        outputs = self.forward_head(x, self.train_index)
        return outputs

    def get_prediction(self, adict, device):
        inputs = adict['inputs'].to(device)
        head_index = adict['head_index']
        outputs = self.forward_head(inputs, head_index)

        probs, outputs = torch.max(torch.softmax(outputs, dim=1), dim=1)

        probs = probs.cpu().detach().numpy()
        classids = outputs.cpu().detach().numpy()

        if self.classnames:
            classnames = [self.classnames[head_index][int(clsid)] for clsid in classids]
        else:
            classnames = []

        return {
            'labels': classids,
            'confidences': probs, 
            'names': classnames,
        }