import torchvision.transforms.functional as TF
import random
import numpy as np
import torch
from PIL import Image
import cv2
from utils.utils import change_box_order, find_intersection, find_jaccard_overlap



class Normalize(object):
        """
        Mean and standard deviation of ImageNet data
        :param mean: (list of float)
        :param std: (list of float)
        """
        def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225],box_transform = True, **kwargs):
            self.mean = mean
            self.std = std
            self.box_transform = box_transform
        def __call__(self, img, box=None, **kwargs):
            """
            :param img: (tensor) image to be normalized
            :param box: (list of tensor) bounding boxes to be normalized, by dividing them with image's width and heights. Format: (x,y,width,height)
            """
            new_img = TF.normalize(img, mean = self.mean, std = self.std)
            if box is not None and self.box_transform:
                _, i_h, i_w = img.size()
                for bb in box:
                    bb[0] = bb[0]*1.0 / i_w
                    bb[1] = bb[1]*1.0 / i_h
                    bb[2] = bb[2]*1.0 / i_w
                    bb[3] = bb[3]*1.0 / i_h

            results = {
                'img': new_img,
                'box': box,
                'label': kwargs['label'],
                'mask': None}
    
            return results


class ToPILImage(object):
    """Convert a tensor or an ndarray to PIL Image.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.

    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
            If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
             - If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
             - If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
             - If the input has 2 channels, the ``mode`` is assumed to be ``LA``.
             - If the input has 1 channel, the ``mode`` is determined by the data type (i.e ``int``, ``float``,
               ``short``).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    """
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.

        Returns:
            PIL Image: Image converted to PIL Image.

        """
        return TF.to_pil_image(pic, self.mode)


class Denormalize(object):
        """
        Denormalize image and boxes for visualization
        """
        def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225],box_transform=True, **kwargs):
            self.mean = mean
            self.std = std
            self.box_transform = box_transform
        def __call__(self, img, box = None, **kwargs):
            """
            :param img: (tensor) image to be denormalized
            :param box: (list of tensor) bounding boxes to be denormalized, by multiplying them with image's width and heights. Format: (x,y,width,height)
            """
            mean = np.array(self.mean)
            std = np.array(self.std)
            img_show = img.numpy().squeeze().transpose((1,2,0))
            img_show = (img_show * std+mean)
            img_show = np.clip(img_show,0,1)


            if box is not None and self.box_transform:
                _, i_h, i_w = img.size()
                for bb in box:
                    bb[0] = bb[0]* i_w
                    bb[1] = bb[1]* i_h
                    bb[2] = bb[2]* i_w
                    bb[3] = bb[3]* i_h

            results = {
                'img': img_show,
                'box': box,
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
            """
            :param img: (PIL Image) image to be tensorized
            :param box: (list of float) bounding boxes to be tensorized. Format: (x,y,width,height)
            :param label: (int) bounding boxes to be tensorized. Format: (x,y,width,height)
            """

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
                box = torch.as_tensor(kwargs['box'], dtype=torch.float32)
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

class Rotation(object):
    '''
        Source: https://github.com/Paperspace/DataAugmentationForObjectDetection
        Rotate image and bounding box
        - Argument:
                    + img: PIL Image
                    + box: list of bounding box for each objects in the image
                    + size: image new size
    '''
    def __init__(self, angle=10):
        self.angle = angle
        if not type(self.angle) == tuple:
            self.angle = (-self.angle, self.angle)


    def rotate_im(self,image, angle):
        """Rotate the image.
        
        Rotate the image such that the rotated image is enclosed inside the tightest
        rectangle. The area not occupied by the pixels of the original image is colored
        black. 
        
        Parameters
        ----------
        
        image : numpy.ndarray
            numpy image
        
        angle : float
            angle by which the image is to be rotated
        
        Returns
        -------
        
        numpy.ndarray
            Rotated Image
        
        """
        # grab the dimensions of the image and then determine the
        # centre
        (h, w) = image.height, image.width
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        image = cv2.warpAffine(np.array(image), M, (nW, nH))

    #    image = cv2.resize(image, (w,h))
        return image


    def bbox_area(self,bbox):
        return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])
    
    def get_corners(self, bboxes):
        """Get corners of bounding boxes

        Parameters
        ----------

        bboxes: numpy.ndarray
            Numpy array containing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`

        returns
        -------

        numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      

        """
        width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
        height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)

        x1 = bboxes[:,0].reshape(-1,1)
        y1 = bboxes[:,1].reshape(-1,1)

        x2 = x1 + width
        y2 = y1 

        x3 = x1
        y3 = y1 + height

        x4 = bboxes[:,2].reshape(-1,1)
        y4 = bboxes[:,3].reshape(-1,1)

        corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))

        return corners

    def rotate_box(self, corners,angle,  cx, cy, h, w):
        """Rotate the bounding box.
        
        
        Parameters
        ----------
        
        corners : numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
        
        angle : float
            angle by which the image is to be rotated
            
        cx : int
            x coordinate of the center of image (about which the box will be rotated)
            
        cy : int
            y coordinate of the center of image (about which the box will be rotated)
            
        h : int 
            height of the image
            
        w : int 
            width of the image
        
        Returns
        -------
        
        numpy.ndarray
            Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
        """

        corners = corners.reshape(-1,2)
        corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
        
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        calculated = np.dot(M,corners.T).T
        
        calculated = calculated.reshape(-1,8)
        
        return calculated

    def get_enclosing_box(self, corners):
        """Get an enclosing box for ratated corners of a bounding box
        
        Parameters
        ----------
        
        corners : numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
        
        Returns 
        -------
        
        numpy.ndarray
            Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`
            
        """
        x_ = corners[:,[0,2,4,6]]
        y_ = corners[:,[1,3,5,7]]
        
        xmin = np.min(x_,1).reshape(-1,1)
        ymin = np.min(y_,1).reshape(-1,1)
        xmax = np.max(x_,1).reshape(-1,1)
        ymax = np.max(y_,1).reshape(-1,1)
        
        final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
        
        return final

    def clip_box(self, bbox, clip_box, alpha):
        """Clip the bounding boxes to the borders of an image

        Parameters
        ----------

        bbox: numpy.ndarray
            Numpy array containing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`

        clip_box: numpy.ndarray
            An array of shape (4,) specifying the diagonal co-ordinates of the image
            The coordinates are represented in the format `x1 y1 x2 y2`

        alpha: float
            If the fraction of a bounding box left in the image after being clipped is 
            less than `alpha` the bounding box is dropped. 

        Returns
        -------

        numpy.ndarray
            Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes left are being clipped and the bounding boxes are represented in the
            format `x1 y1 x2 y2` 

        """
        ar_ = (self.bbox_area(bbox))
        x_min = np.maximum(bbox[:,0], clip_box[0]).reshape(-1,1)
        y_min = np.maximum(bbox[:,1], clip_box[1]).reshape(-1,1)
        x_max = np.minimum(bbox[:,2], clip_box[2]).reshape(-1,1)
        y_max = np.minimum(bbox[:,3], clip_box[3]).reshape(-1,1)

        bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:,4:]))

        delta_area = ((ar_ - self.bbox_area(bbox))/ar_)

        mask = (delta_area < (1 - alpha)).astype(int)

        bbox = bbox[mask == 1,:]
        return bbox

    def __call__(self, img, box = None, **kwargs):
        
        
        angle = random.uniform(*self.angle)
        w,h = img.width, img.height
        cx, cy = w//2, h//2
        img = self.rotate_im(img, angle)
        
        if box is not None:
            new_box = change_box_order(box, 'xywh2xyxy')
            corners = self.get_corners(new_box)
            corners = np.hstack((corners, new_box[:,4:]))
            corners[:,:8] = self.rotate_box(corners[:,:8], angle, cx, cy, h, w)
            new_bbox = self.get_enclosing_box(corners)
            scale_factor_x = img.shape[1] / w
            scale_factor_y = img.shape[0] / h
            img = cv2.resize(img, (w,h))
            new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
            new_box = new_bbox
            new_box = self.clip_box(new_box, [0,0,w, h], 0.25)
            new_box = change_box_order(new_box, 'xyxy2xywh')
        else:
            new_box = box
        
        img=Image.fromarray(img)

        return {
            'img': img, 
            'box': new_box,
            'label': kwargs['label'],
            'mask': None}

class RandomHorizontalFlip(object):
        """
        Horizontally flip image and its bounding box, mask
        """
        def __init__(self, ratio = 0.5):
            self.ratio = ratio
          
        def __call__(self, img, box = None, **kwargs):
            if random.randint(1,10) <= self.ratio*10:
                # Flip image
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

                # Flip bounding box
                if box is not None:
                    new_box = change_box_order(box, 'xywh2xyxy')
                    w = img.width
                    xmin = w - new_box[:,2]
                    xmax = w - new_box[:,0]
                    new_box[:,0] = xmin
                    new_box[:,2] = xmax
                    new_box = change_box_order(new_box, 'xyxy2xywh')
                else:
                    new_box = box
            else:
                new_box = box

            results = {
                'img': img,
                'box': new_box,
                'label': kwargs['label'],
                'mask': None}
    
            return results

class RandomCrop(object):
    """
    Source: https://github.com/anhtuan85/Data-Augmentation-for-Object-Detection
    """
    def __init__(self):
        self.ratios = [0.3, 0.5, 0.9, None]
        
    def __call__(self, img, box = None, **kwargs):
        '''
        image: A PIL image
        boxes: Bounding boxes, a tensor of dimensions (#objects, 4)
        labels: labels of object, a tensor of dimensions (#objects)
        difficulties: difficulties of detect object, a tensor of dimensions (#objects)
        
        Out: cropped image , new boxes, new labels, new difficulties
        '''
      
        image = TF.to_tensor(img)
        original_h = image.size(1)
        original_w = image.size(2)

        while True:
            mode = random.choice(self.ratios)

            if mode is None:
                return {
                    'img': img,
                    'box': box,
                    'label': kwargs['label'],
                    'mask': None}

            boxes = change_box_order(box, 'xywh2xyxy')
            
            boxes = torch.FloatTensor(boxes)
            labels = torch.LongTensor(kwargs['label'])
                
            new_image = image
            new_boxes = boxes
            new_labels = labels

            for _ in range(50):
                # Crop dimensions: [0.3, 1] of original dimensions
                new_h = random.uniform(0.3*original_h, original_h)
                new_w = random.uniform(0.3*original_w, original_w)

                # Aspect ratio constraint b/t .5 & 2
                if new_h/new_w < 0.5 or new_h/new_w > 2:
                    continue

                #Crop coordinate
                left = random.uniform(0, original_w - new_w)
                right = left + new_w
                top = random.uniform(0, original_h - new_h)
                bottom = top + new_h
                crop = torch.FloatTensor([int(left), int(top), int(right), int(bottom)])

                # Calculate IoU  between the crop and the bounding boxes
                overlap = find_jaccard_overlap(crop.unsqueeze(0), boxes) #(1, #objects)
                overlap = overlap.squeeze(0)
                # If not a single bounding box has a IoU of greater than the minimum, try again
                if overlap.max().item() < mode:
                    continue

                #Crop
                new_image = image[:, int(top):int(bottom), int(left):int(right)] #(3, new_h, new_w)

                #Center of bounding boxes
                center_bb = (boxes[:, :2] + boxes[:, 2:])/2.0

                #Find bounding box has been had center in crop
                center_in_crop = (center_bb[:, 0] >left) * (center_bb[:, 0] < right
                                 ) *(center_bb[:, 1] > top) * (center_bb[:, 1] < bottom)    #( #objects)

                if not center_in_crop.any():
                    continue

                #take matching bounding box
                new_boxes = boxes[center_in_crop, :]

                #take matching labels
                new_labels = labels[center_in_crop]

                #Use the box left and top corner or the crop's
                new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])

                #adjust to crop
                new_boxes[:, :2] -= crop[:2]

                new_boxes[:, 2:] = torch.min(new_boxes[:, 2:],crop[2:])

                #adjust to crop
                new_boxes[:, 2:] -= crop[:2]
                
                
                new_boxes = change_box_order(new_boxes, 'xyxy2xywh')
                return {
                        'img': TF.to_pil_image(new_image),
                        'box': new_boxes.numpy(),
                        'label': new_labels.numpy(),
                        'mask': None}


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
            
            
            if transforms_list is None:
                self.transforms_list = [Resize(), ToTensor(), Normalize()]
            else:
              self.transforms_list = transforms_list
            if not isinstance(self.transforms_list,list):
                self.transforms_list = list(self.transforms_list)
            
            for x in self.transforms_list:
                if isinstance(x, Normalize):
                    self.denormalize = Denormalize(box_transform=x.box_transform)

        def __call__(self, img, box = None, label = None, mask = None):
            for tf in self.transforms_list:
                results = tf(img = img, box = box, label = label, mask = mask)
                img = results['img']
                box = results['box']
                label = results['label']
                mask = results['mask']

            return results