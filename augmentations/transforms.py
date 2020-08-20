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
            
            results = {
                'img': new_img,
                'box': kwargs['box'],
                'label': kwargs['label'],
                'mask': None}
    
            return results

class Denormalize(object):
        """
        Denormalize image and numpify all for visualization
        """
        def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], **kwargs):
            self.mean = mean
            self.std = std
        def __call__(self, img, **kwargs):
            mean = np.array(self.mean)
            std = np.array(self.std)
            img_show = img.numpy().squeeze().transpose((1,2,0))
            img_show = (img_show * std+mean)
            img_show = np.clip(img_show,0,1)

            results = {
                'img': img_show,
                'box': kwargs['box'],
                'label': kwargs['label'],
                'mask': None}
    
            return results
           

class ToTensor(object):
        """
        Tensorize image
        """
        def __init__(self):
            pass
        def __call__(self, img, **kwargs):
            img = TF.to_tensor(img)
            
            results = {
                'img': img,
                'box': kwargs['box'],
                'label': kwargs['label'],
                'mask': None}

            if kwargs['label'] is not None:
                label = torch.LongTensor(kwargs['label'])
                results['label'] = label
            if kwargs['box'] is not None:
                box = torch.FloatTensor(kwargs['box'])
                results['box'] = box

            return results
           

class Resize(object):
        """
        - Resize an image and bounding box, mask
        - Argument:
                    + img: PIL Image
                    + box: list of bounding box for each objects in the image
                    + size: image new size
        """
        def __init__(self, size = (224,224), **kwargs):
            self.size = size

        def __call__(self, img, box = None,  **kwargs):
            # Resize image
            new_img = TF.resize(img, size=self.size)

            if box is not None:
                np_box = np.array(box)
                old_dims = np.array([img.width, img.height, img.width, img.height])
                new_dims = np.array([self.size[1], self.size[0], self.size[1], self.size[0]])

                # Resize bounding box and round down
                box = np.floor((np_box / old_dims) * new_dims)

            results = {
                'img': new_img,
                'box': box,
                'label': kwargs['label'],
                'mask': None}
    
            return results
            


class RandomHorizontalFlip(object):
        """
        Horizontally flip image and its bounding box, mask
        """
        def __init__(self, ratio = 0.5):
            self.ratio = ratio
          
        def __call__(self, img, box = None, **kwargs):
            if random.randint(1,10) <= self.ratio*10:
                # Flip image
                img = TF.hflip(img)

                # Flip bounding box
                if box is not None:
                    img_center = np.array(np.array(img).shape[:2])/2
                    img_center = np.hstack((img_center, img_center))
                    box[:, [0, 2]] += 2*(img_center[[0, 2]] - box[:, [0, 2]])
                    box_w = abs(box[:, 0] - box[:, 2])
                    box[:, 0] -= box_w
                    box[:, 2] += box_w
     
            results = {
                'img': img,
                'box': box,
                'label': kwargs['label'],
                'mask': None}
    
            return results
            

class Compose(object):
        """
        - Custom Transform class include image augmentation methods, return dict
        - Can apply for all tasks
        - Examples:
                    my_transforms = Compose(transforms_list=[
                                                Resize((300,300)),
                                                #RandomHorizontalFlip(),
                                                ToTensor(),
                                                Normalize()])
                    results = my_transforms(img, box, label)
                    img, box, label, mask = results['img'], results['box'], results['label'], results['mask']
        """
        def __init__(self, transforms_list = None):
            self.denormalize = Denormalize()
            
            if transforms_list is None:
                self.transforms_list = [Resize(), ToTensor(), Normalize()]
            else:
              self.transforms_list = transforms_list
            if not isinstance(self.transforms_list,list):
                self.transforms_list = list(self.transforms_list)
                
        def __call__(self, img, box = None, label = None, mask = None):
            for tf in self.transforms_list:
                results = tf(img = img, box = box, label = label, mask = mask)
                img = results['img']
                box = results['box']
                label = results['label']
                mask = results['mask']

            return results