import torchvision.transforms.functional as TF
import random
import numpy as np
import torch
from PIL import Image

class Normalize(object):
        """
        Mean and standard deviation of ImageNet data
        :param mean: (list of float)
        :param std: (list of float)
        """
        def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], box_transforms = True, **kwargs):
            self.mean = mean
            self.std = std
            self.box_transforms = box_transforms
        def __call__(self, img, box=None, **kwargs):
            """
            :param img: (tensor) image to be normalized
            :param box: (list of tensor) bounding boxes to be normalized, by dividing them with image's width and heights. Format: (x,y,width,height)
            """
            new_img = TF.normalize(img, mean = self.mean, std = self.std)
            if box is not None and self.box_transforms:
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

class Denormalize(object):
        """
        Denormalize image and boxes for visualization
        """
        def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], **kwargs):
            self.mean = mean
            self.std = std
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


            if box is not None:
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


    def rotate_im(self, image, angle):
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
        (h, w) = image.shape[:2]
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
        image = cv2.warpAffine(image, M, (nW, nH))

    #    image = cv2.resize(image, (w,h))
        return image



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


    def __call__(self, img, box = None, **kwargs):

        angle = random.uniform(*self.angle)
        w,h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2
        img = self.rotate_im(img, angle)
        corners = self.get_corners(box)
        corners = np.hstack((corners, box[:,4:]))
        corners[:,:8] = self.rotate_box(corners[:,:8], angle, cx, cy, h, w)
        new_bbox = self.get_enclosing_box(corners)
        scale_factor_x = img.shape[1] / w
        scale_factor_y = img.shape[0] / h
        img = cv2.resize(img, (w,h))
        new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
        bboxes  = new_bbox
        bboxes = clip_box(bboxes, [0,0,w, h], 0.25)
        return {
            'img': img, 
            'box': bboxes,
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
                    w = img.width
                    xmin = w - box[:,2]
                    xmax = w - box[:,0]
                    boxes[:,0] = xmin
                    boxes[:,2] = xmax
                
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