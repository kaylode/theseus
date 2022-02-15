from typing import Dict, List, Optional
import os
import pandas as pd
from .dataset import SegmentationDataset

class CSVDataset(SegmentationDataset):
    r"""CSVDataset multi-labels segmentation dataset

    Reads in .csv file with structure below:
        filename   | label
        ---------- | -----------
        <img1>.jpg | <mask1>.jpg

    image_dir: `str`
        path to directory contains images
    mask_dir: `str`
        path to directory contains masks
    transform: Optional[List]
        transformatin functions
        
    """
    def __init__(
            self, 
            image_dir: str, 
            mask_dir: str, 
            csv_path: str, 
            txt_classnames: str,
            transform: Optional[List] = None,
            **kwargs):
        super(CSVDataset, self).__init__(**kwargs)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.csv_path = csv_path
        self.transform = transform
        self.txt_classnames = txt_classnames
        self._load_data()

    def _load_data(self):
        """
        Read data from csv and load into memory
        """

        with open(self.txt_classnames, 'r') as f:
            self.classnames = f.read().splitlines()
        
        # Mapping between classnames and indices
        for idx, classname in enumerate(self.classnames):
            self.classes_idx[classname] = idx
        self.num_classes = len(self.classnames)
        
        df = pd.read_csv(self.csv_path)
        for idx, row in df.iterrows():
            img_name, mask_name = row
            image_path = os.path.join(self.image_dir,img_name)
            mask_path = os.path.join(self.mask_dir, mask_name)
            self.fns.append([image_path, mask_path])
