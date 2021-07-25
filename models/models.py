import timm
import torch.nn as nn
import torch.nn.functional as F

class BaseTimmModel(nn.Module):
    """Some Information about BaseTimmModel"""

    def __init__(
        self,
        num_classes=1000,
        name="vit_base_patch16_224",
        from_pretrained=True
    ):
        super().__init__()
        self.name = name
        if num_classes != 1000:
            self.model = timm.create_model(name, pretrained=from_pretrained, num_classes=num_classes)
        else:
            self.model = timm.create_model(name, pretrained=from_pretrained)

        self.model = nn.DataParallel(self.model)

    def forward(self, batch, device):
        inputs = batch["imgs"]
        inputs = inputs.to(device)
        outputs = self.model(inputs)
        return outputs