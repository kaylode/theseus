import os
import pandas as pd
from typing import List, Optional

from torchvision.transforms import transforms as tf
from theseus.classification.augmentations.custom import RandomMixup, RandomCutmix
from theseus.utilities.loggers.observer import LoggerObserver
from .dataset import ClassificationDataset

LOGGER = LoggerObserver.getLogger('main')

class ImageFolderDataset(ClassificationDataset):
    r"""ImageFolderDataset multi-labels classification dataset

    Reads in folder of images with structure below:
        |<classname1>
            |--- <image1>.jpg
            |--- <image2>.jpg
        |<classname2>
            |--- <image3>.jpg
            |--- <image4>.jpg

    image_dir: `str`
        path to directory contains images
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
        txt_classnames: str,
        transform: Optional[List] = None,
        test: bool = False,
        **kwargs
    ):
        super(ImageFolderDataset, self).__init__(test, **kwargs)
        self.image_dir = image_dir
        self.txt_classnames = txt_classnames
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

        # Get classnames
        with open(self.txt_classnames, 'r') as f:
            self.classnames = f.read().splitlines()
        
        # Mapping between classnames and indices
        for idx, classname in enumerate(self.classnames):
            self.classes_idx[classname] = idx
        self.num_classes = len(self.classnames)

        classnames = os.listdir(self.image_dir)
        for label in classnames:
            folder_name = os.path.join(self.image_dir, label)
            image_names = os.listdir(folder_name)
            for image_name in image_names:
                self.fns.append([os.path.join(folder_name, image_name), label])
    
    def _calculate_classes_dist(self):
        """
        Calculate distribution of classes
        """
        LOGGER.text("Calculating class distribution...", LoggerObserver.DEBUG)
        self.classes_dist = []

        classnames = os.listdir(self.image_dir)
        for label in classnames:
            folder_name = os.path.join(self.image_dir, label)
            image_names = os.listdir(folder_name)
            for _ in image_names:
                self.classes_dist.append(self.classes_idx[label])

        return self.classes_dist