from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from theseus.utilities.cuda import move_to
from collections import OrderedDict

# https://huggingface.co/docs/transformers/task_summary

class HuggingFaceModel(nn.Module):
    """Convolution models from timm
    
    model_name: `str`
        huggingface model name
    num_classes: `int`
        number of classes
    from_pretrained: `bool` 
        whether to use pretrained
    classnames: `Optional[List]`
        list of classnames
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 1000,
        from_pretrained: bool = True,
        classnames: Optional[List] = None,
        pooling: str = 'first',
        freeze: bool = False,
        **kwargs
    ):
        super().__init__()
        self.name = model_name
        self.pooling = pooling
        self.freeze = freeze

        self.classnames = classnames

        if from_pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_config(config)

        self.feature_dim = self.model.config.hidden_size

        self.head = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(self.feature_dim, num_classes) if num_classes > 0 else nn.Identity())
        ]))

        if self.freeze:
            self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def get_model(self):
        """
        Return the full architecture of the model, for visualization
        """
        return self.model

    def forward_features(self, batch: Dict, device: torch.device):

        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        transformer_out = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        features = transformer_out.last_hidden_state

        if self.pooling == 'average':
            features = torch.mean(features, dim=1) #features[:,0,:]
        elif self.pooling == 'first':
            features = features[:,0,:]
        else:
            raise ValueError()

        return features

    def forward(self, batch: Dict, device: torch.device):
        batch = move_to(batch, device)
        features = self.forward_features(batch, device)
        outputs = self.head(features)

        return {
            'outputs': outputs,
            'features': features
        }


    def get_prediction(self, adict: Dict[str, Any], device: torch.device):
        """
        Inference using the model.

        adict: `Dict[str, Any]`
            dictionary of inputs
        device: `torch.device`
            current device 
        """
        outputs = self.model(adict, device)['outputs']

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
