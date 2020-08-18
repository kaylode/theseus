import torchvision.transforms.functional as TF
import random
import numpy as np
import torch

class Normalize(object):
        """
        Mean and standard deviation of ImageNet data
        """
        def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], **kwargs):
            self.mean = mean
            self.std = std
        def __call__(self, img, **kwargs):
            new_img = TF.normalize(img, mean = self.mean, std = self.std)
            return new_img, kwargs["bboxes"], kwargs["classes"]

class Denormalize(object):
        """
        Denormalize image and numpify all for visualization
        """
        def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], **kwargs):
            self.mean = mean
            self.std = std
        def __call__(self, img, bboxes, classes):
            mean = np.array(self.mean)
            std = np.array(self.std)
            img_show = img.numpy().squeeze().transpose((1,2,0))
            img_show = (img_show * std+mean)
            img_show = np.clip(img_show,0,1)
            return img_show, bboxes.numpy(), classes.numpy()

class ToTensor(object):
        """
        Tensorize
        """
        def __init__(self):
            pass
        def __call__(self, img, bboxes, classes, **kwargs):
            img_tensor = TF.to_tensor(img)
            bboxes_tensor = torch.FloatTensor(bboxes)
            classes_tensor = torch.LongTensor(classes)
            return img_tensor, bboxes_tensor, classes_tensor

class Resize(object):
        """
        - Resize an image and bounding boxes
        - Argument:
                    + img: PIL Image
                    + bboxes: list of bounding boxes for each objects in the image
                    + size: image new size
        """
        def __init__(self, size = (224,224), **kwargs):
            self.size = size

        def __call__(self, img, bboxes, **kwargs):
            # Resize image
            new_img = TF.resize(img, size=self.size)

            np_bboxes = np.array(bboxes)
            old_dims = np.array([img.width, img.height, img.width, img.height])
            new_dims = np.array([self.size[1], self.size[0], self.size[1], self.size[0]])

            # Resize bounding boxes and round down
            new_bboxes = np.floor((np_bboxes / old_dims) * new_dims)
            return new_img, new_bboxes, kwargs["classes"]


class RandomHorizontalFlip(object):
        """
        Horizontally flip image and its bounding boxes
        """
        def __init__(self, ratio = 0.5):
            self.ratio = ratio
          
        def __call__(self, img, bboxes, **kwargs):
            if random.randint(1,10) <= self.ratio*10:
                # Flip image
                img = TF.hflip(img)

                # Flip bounding boxes

                img_center = np.array(np.array(img).shape[:2])/2
                img_center = np.hstack((img_center, img_center))

        
                bboxes[:, [0, 2]] += 2*(img_center[[0, 2]] - bboxes[:, [0, 2]])

                box_w = abs(bboxes[:, 0] - bboxes[:, 2])

                bboxes[:, 0] -= box_w
                bboxes[:, 2] += box_w
     
            return img, bboxes, kwargs['classes']

class Compose(object):
        """
        - Custom Transform class include image augmentation methods
        - Can apply for all tasks
        - Examples:
                    my_transforms = Compose(transforms_list=[
                                                Resize((300,300)),
                                                #RandomHorizontalFlip(),
                                                ToTensor(),
                                                Normalize()])
                    img, boxes, classes = my_transforms(img, boxes, classes)
        """
        def __init__(self, transforms_list = None):
            self.denormalize = Denormalize()
            
            if transforms_list is None:
                self.transforms_list = [Resize(), ToTensor(), Normalize()]
            else:
              self.transforms_list = transforms_list
            if not isinstance(self.transforms_list,list):
                self.transforms_list = list(self.transforms_list)
                
        def __call__(self, img, bboxes, classes):
            for tf in self.transforms_list:
                img, bboxes, classes = tf(img = img, bboxes = bboxes, classes = classes)
            return img, bboxes, classes