import os
import torch
from PIL import Image
from typing import Dict, List

class DetectionDataset(torch.utils.data.Dataset):
    r"""Base dataset for classification tasks
    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        self.classes_idx = {}
        self.classnames = None
        self.transform = None
        self.fns = [] 

    def _load_data(self):
        raise NotImplementedError
    
    def load_image_and_boxes(self, index):
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict:
        """
        Get one item
        """
        (image, boxes, labels, img_id, 
            img_name, ori_width, ori_height) = self.load_image_and_boxes(idx)


        if self.transform:
            item = self.transform(image=image, bboxes=boxes, class_labels=labels)
            # Normalize
            image = item['image']
            boxes = item['bboxes']
            labels = item['class_labels']

        if len(boxes) == 0:
            return self.__getitem__((idx+1)%len(self.image_ids))

        labels = torch.LongTensor(labels) # starts from 1
        boxes = torch.as_tensor(boxes, dtype=torch.float32) 

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        return {
            'img': image,
            'target': target,
            'img_id': img_id,
            'img_name': img_name,
            'ori_size': [ori_width, ori_height]
        }

    def __len__(self) -> int:
        return len(self.fns)

    def collate_fn(self, batch: List):
        """
        Collator for wrapping a batch
        """
        raise NotImplementedError