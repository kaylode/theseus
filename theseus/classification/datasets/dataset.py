import os
import pandas as pd
from PIL import Image
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torchvision.transforms import transforms as tf

from theseus.classification.augmentations.custom import RandomMixup, RandomCutmix

class CSVDataset(torch.utils.data.Dataset):
    r"""CSVDataset multi-labels classification dataset

    image_dir: `str`
        path to directory contains images
    csv_path: `str`
        path to csv file
    txt_classnames: `str`
        path to txt file contains classnames
    transform: Optional[List]
        transformatin functions
    test: bool
        whether the dataset is used for training or test
        
    """

    def __init__(
        self,
        image_dir: str,
        csv_path: str,
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
        """
        Read data from csv and load into memory
        """
        self.fns = []

        # Classes distribution (for balanced sampler)
        self.classes_dist = []

        # Get classnames
        self.classes_idx = {}
        with open(self.txt_classnames, 'r') as f:
            self.classnames = f.read().splitlines()
        
        # Mapping between classnames and indices
        for idx, classname in enumerate(self.classnames):
            self.classes_idx[classname] = idx
        self.num_classes = len(self.classnames)

        # Load csv
        df = pd.read_csv(self.csv_path)
        for _, row in df.iterrows():
            image_name, label = row
            image_path = os.path.join(self.image_dir, image_name)
            self.fns.append([image_path, label])
            self.classes_dist.append(self.classes_idx[label])

    def __getitem__(self, idx: int) -> Dict:
        """
        Get one item
        """
        image_path, label_name = self.fns[idx]
        im = Image.open(image_path).convert('RGB')
        width, height = im.width, im.height
        class_idx = self.classes_idx[label_name]
        transformed = self.transform(im)

        target = {}
        target['labels'] = [class_idx]
        target['label_name'] = label_name

        return {
            "input": transformed, 
            'target': target,
            'img_name': os.path.basename(image_path),
            'ori_size': [width, height]
        }

    def __len__(self) -> int:
        return len(self.fns)

    def collate_fn(self, batch: List):
        """
        Collator for wrapping a batch
        """
        imgs = torch.stack([s['input'] for s in batch])
        targets = torch.stack([torch.LongTensor(s['target']['labels']) for s in batch])

        # if self.mixupcutmix is not None:
        #     imgs, targets = self.mixupcutmix(imgs, targets.squeeze(1))
        # targets = targets.float()

        return {
            'inputs': imgs,
            'targets': targets
        }