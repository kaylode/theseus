import torch
import numpy as np
from albumentations.core.transforms_interface import DualTransform

class BoxNormalize(DualTransform):
    '''
    Bounding boxes normalization
    '''

    def __init__(self, order):
        super().__init__(always_apply=True, p=1.0)
        self.order = order
    
    def apply(self, image, **params):
        return image

    def apply_to_bbox(self, bbox, **params):
        """
        Normalize bbox
        :return: (tensor) converted bounding bbox, size [N, 4]
        """
        height = params["rows"]
        width = params["cols"]

        bbox = list(bbox)
        if self.order in ['xywh', 'cxcywh']:
            bbox[0] /= width 
            bbox[2] /= width 
            bbox[1] /= height
            bbox[3] /= height
        return tuple(bbox)

    def get_transform_init_args_names(self):
        """
        Fetches the parameter(s) of __init__ method
        :returns: tuple of parameter(s) of __init__ method
        """
        return ('order', 'always_apply', 'p')

class BoxOrder(DualTransform):
    """
    Bounding bbox reorder
    """
    
    def __init__(
        self,
        order
    ):
        """
        Class construstor
        :param order: bbox format
        """
        super(BoxOrder, self).__init__(always_apply=True, p=1.0)  # Initialize parent class
        self.order = order
        
    def apply(self, image, **params):
        """
        Applies the reorder augmentation on the given image
        
        :param image: The image to be augmented
        :returns augmented image
        """
        return image

    def apply_to_bbox(self, bbox, **params):

        """
        Change box order between (xmin, ymin, xmax, ymax) and (xcenter, ycenter, width, height).
        :param bbox: (tensor) or {np.array) bounding bbox, sized [N, 4]
        :param order: (str) ['xyxy2xywh', 'xywh2xyxy', 'xyxy2cxcywh', 'cxcywh2xyxy']
        :return: (tensor) converted bounding bbox, size [N, 4]
        """

        assert self.order   in ['xyxy2xywh', 'xywh2xyxy', 'xyxy2cxcywh', 'cxcywh2xyxy', 'xywh2cxcywh', 'cxcywh2xywh']
        if self.order == 'xywh2xyxy':
            new_bbox = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
        elif self.order == 'xyxy2xywh':
            new_bbox = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
        elif self.order == 'xyxy2cxcywh':
            new_bbox = (
                (bbox[0] + bbox[2]) / 2, 
                (bbox[1] + bbox[3]) / 2,
                (bbox[2] - bbox[0]), 
                (bbox[3] - bbox[1])
            )
        elif self.order == 'xywh2cxcywh':
            new_bbox = (
                bbox[0] + (bbox[2] / 2), 
                bbox[1] + (bbox[3] / 2),
                bbox[2], bbox[3]
            )
        elif self.order == 'cxcywh2xyxy':
            new_bbox = (
                bbox[0] - (bbox[2] / 2), 
                bbox[1] - (bbox[3] / 2),
                bbox[0] + (bbox[2] / 2), 
                bbox[1] + (bbox[3] / 2), 
            )
        elif self.order == 'cxcywh2xywh':
            new_bbox = (
                bbox[0] - (bbox[2] / 2), 
                bbox[1] - (bbox[3] / 2),
                bbox[2], bbox[3]
            )
        return new_bbox

    def get_transform_init_args_names(self):
        """
        Fetches the parameter(s) of __init__ method
        :returns: tuple of parameter(s) of __init__ method
        """
        return ('order',  'always_apply', 'p')