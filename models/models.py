import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseTimmModel(nn.Module):
    """Some Information about BaseTimmModel"""

    def __init__(
        self,
        num_classes,
        name="vit_base_patch16_224",
        from_pretrained=True,
        freeze_backbone=False,
    ):
        super().__init__()
        self.name = name
        self.model = timm.create_model(name, pretrained=from_pretrained)
        if name.find("nfnet") != -1:
            self.model.head.fc = nn.Linear(self.model.head.fc.in_features, num_classes)
        elif name.find("efficientnet") != -1:
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features, num_classes
            )
        elif name.find("resnext") != -1:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif name.find("vit") != -1:
            self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        elif name.find("densenet") != -1:
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features, num_classes
            )
        else:
            assert False, "Classifier block not included in TimmModel"

        self.model = nn.DataParallel(self.model)

    def forward(self, batch, device):
        inputs = batch["imgs"]
        inputs = inputs.to(device)
        outputs = self.model(inputs)
        return outputs