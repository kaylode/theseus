from theseus.base.models import MODEL_REGISTRY

from .unet import U_Net, R2U_Net, R2AttU_Net, AttU_Net, NestedUNet
from .wrapper import ModelWithLoss

MODEL_REGISTRY.register(U_Net)
MODEL_REGISTRY.register(R2U_Net)
MODEL_REGISTRY.register(R2AttU_Net)
MODEL_REGISTRY.register(AttU_Net)
MODEL_REGISTRY.register(NestedUNet)