import torch
from torchvision.transforms import transforms as tf
from theseus.base.datasets.collator import BaseCollator
from theseus.classification.augmentations.custom import RandomMixup, RandomCutmix


class MixupCutmixCollator(BaseCollator):
    """Apply mixup and cutmix to a batch, temporarily supports classification only
    """
    def __init__(
        self, 
        dataset: torch.utils.data.Dataset, 
        mixup_alpha=0.2, cutmix_alpha=1.0, 
        weight=[0.5, 0.5], **kwargs) -> None:

        mixup_transforms = []
        mixup_transforms.append(RandomMixup(dataset.num_classes, p=1.0, alpha=mixup_alpha))
        mixup_transforms.append(RandomCutmix(dataset.num_classes, p=1.0, alpha=cutmix_alpha))
        self.mixupcutmix = tf.RandomChoice(mixup_transforms, p=weight)

    def __call__(self, batch):
        imgs, targets = self.mixupcutmix(
            batch['inputs'], batch['targets'].squeeze(1))
        batch['inputs'] = imgs
        batch['targets'] = targets
        return batch