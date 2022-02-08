import os
import pandas as pd
from PIL import Image
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torchvision.transforms import transforms as tf
from torckay.base.datasets import DATASET_REGISTRY

from torckay.classification.augmentations.custom import RandomMixup, RandomCutmix

@DATASET_REGISTRY.register()
class CSVDataset(torch.utils.data.Dataset):
    r"""CSVDataset multi-labels classification dataset


    Attributes:
        from_list(**args): Create dataset from list
        from_folder(**args): Create dataset from folder path

    """

    def __init__(
        self,
        image_dir: List[str],
        csv_path: List[str],
        txt_classnames: str,
        transform: Optional[List] = None,
        test: bool = False,
    ):
        super(CSVDataset, self).__init__()
        self.image_dir = image_dir
        self.txt_classnames = txt_classnames
        self.csv_path = csv_path
        self.train = not (test)
        self.transform = transform
        self._load_data()

        if self.train:
            # MixUp and CutMix
            mixup_transforms = []
            mixup_transforms.append(RandomMixup(self.num_classes, p=1.0, alpha=0.2))
            mixup_transforms.append(RandomCutmix(self.num_classes, p=1.0, alpha=1.0))
            self.mixupcutmix = tf.RandomChoice(mixup_transforms)
        else:
            self.mixupcutmix = None

    def _load_data(self):
        self.fns = []
        self.classes_dist = []

        self.classes_idx = {}
        with open(self.txt_classnames, 'r') as f:
            self.classnames = f.read().splitlines()
        
        for idx, classname in enumerate(self.classnames):
            self.classes_idx[classname] = idx
        self.num_classes = len(self.classnames)
        df = pd.read_csv(self.csv_path)
        for _, row in df.iterrows():
            image_name, label = row
            image_path = os.path.join(self.image_dir, image_name)
            self.fns.append([image_path, self.classes_idx[label]])
            self.classes_dist.append(self.classes_idx[label])

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:

        image_path, class_idx = self.fns[idx]
        im = Image.open(image_path).convert('RGB')
        label = torch.LongTensor([class_idx])  # convert label to 0 - 1 (W, H)

        transformed = self.transform(im)

        item = {"input": transformed, "target": label.long()}
        return item

    def __len__(self) -> int:
        return len(self.fns)

    def collate_fn(self, batch: List):
        imgs = torch.stack([s['input'] for s in batch])
        targets = torch.stack([s['target'] for s in batch])

        # if self.mixupcutmix is not None:
        #     imgs, targets = self.mixupcutmix(imgs, targets.squeeze(1))
        # targets = targets.float()

        return {
            'input': imgs,
            'target': targets
        }

    
    @classmethod
    def from_folder(
        cls,
        image_dir: List[str],
        csv_path: List[str],
        txt_classnames: str,
        test: bool = False,
        transform: Optional[List] = None,
    ):
        r"""From folder method

        Args:
            root: folder root
            image_folder_name: image folder name
            mask_folder_name: label folder name
            extension: image file type extenstion. Defaults to "png".
            test: Option using for inference mode. if True, __getitem__ does not return label.
                Defaults to False.
            transform: rgb transform. Defaults to None.

        Returns:
            CSVDataset: dataset class
        """

        return cls(image_dir, csv_path, txt_classnames, test=test, transform=transform)


DATASET_REGISTRY._do_register("CSVDataset.from_folder", CSVDataset.from_folder)

