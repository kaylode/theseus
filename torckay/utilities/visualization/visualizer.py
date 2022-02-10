import cv2
import random
import numpy as np
from PIL import Image
from typing import Optional
from .colors import color_list
from .utils import (
    draw_mask, draw_polylines, draw_text,
    get_font_size, reduce_opacity, draw_text_cv2)

class Visualizer():
    def __init__(self):
        self.image: Optional[np.ndarray] = None
        self.class_names = None

    def set_image(self, image: np.ndarray) -> None:
        self.image = image

        if self.image.dtype == 'uint8':
            self.image = self.image / 255.0
        
    def set_classnames(self, class_names) -> None:
        self.class_names = class_names

    def get_image(self):
        if self.image.dtype == 'uint8':
            self.image = np.clip(self.image, 0, 255)
        elif self.image.dtype == 'float32' or self.image.dtype == 'float64':
            self.image = np.clip(self.image, 0.0, 1.0)
            self.image = (self.image*255).astype(np.uint8)

        return self.image

    def save_image(self, path):
        cv2.imwrite(path, self.get_image()[:,:,::-1])

    def draw_label(self, 
            label,
            font                   = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale              = 2,
            fontColor              = (0,0,1),
            thickness              = 3,
            outline                = (0,0,0),
            offset = 50 
        ):
        assert self.image is not None
        if self.class_names is not None:
            label = self.class_names[label]

        h,w,c = self.image.shape
        white_canvas = np.ones((h+offset,w, c))
        white_canvas[:h,:w,:c] = self.image
        bottomLeftCornerOfText = (int(w/6), h+10)

        draw_text_cv2(white_canvas, str(label), 
            bottomLeftCornerOfText, 
            fontFace=font, 
            fontScale=fontScale,
            color=fontColor,
            outline_color=outline,
            thickness=thickness)

        self.image = white_canvas.copy()


    def draw_polygon_ocr(self, polygons, texts=None, font='assets/fonts/aachenb.ttf'):
        image = self.image.copy()
        maskIm = Image.new('L', (self.image.shape[1], self.image.shape[0]), 0)

        if texts is not None:
            zipped = zip(polygons, texts)
        else:
            zipped = polygons

        for item in zip(zipped): 
            if texts is not None:
                polygon, text = item
            else:
                polygon, text = item, None

            maskIm = draw_mask(polygon, maskIm) 
            image = draw_polylines(image, polygon)

            if text:
                font_size = get_font_size(image, text, polygon, font)
                color = tuple([random.randint(0,255)/255.0 for _ in range(3)])
                white_img = draw_text(white_img, text, polygon, font, color, font_size)

        # Mask out polygons
        mask = np.stack([maskIm, maskIm, maskIm], axis=2)
        masked = image * mask

        # Reduce opacity of original image
        o_image = reduce_opacity(image)
        i_masked = (np.bitwise_not(mask)/255).astype(np.int)
        o_image = o_image * i_masked

        # Add two image
        new_img = o_image + masked

        new_img = new_img.astype(np.float32)

        if text:
            white_img = white_img.astype(np.float32)
            stacked = np.concatenate([new_img, white_img], axis=1)
            self.image = stacked.copy()
        else:
            self.image = new_img.copy()


    def draw_bbox(self, boxes, labels=None) -> None:
        assert self.image is not None
        
        tl = int(round(0.001 * max(self.image.shape[:2])))  # line thickness
        
        tup = zip(boxes, labels) if labels is not None else boxes

        for item in tup:
            if labels is not None:
                box, label = item
                color = color_list[int(label)]
            else:
                box, label = item, None

            coord = [box[0], box[1], box[2], box[3]]
            c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
            cv2.rectangle(self.image, c1, c2, color, thickness=tl*2)

            if label is not None:
                if self.class_names is not None:
                    label = self.class_names[label]
                tf = max(tl - 2, 1)  # font thickness
                s_size = cv2.getTextSize(f'{label}', 0, fontScale=float(tl) / 3, thickness=tf)[0]
                c2 = c1[0] + s_size[0] + 15, c1[1] - s_size[1] - 3
                cv2.rectangle(self.image, c1, c2, color, -1)  # filled
                cv2.putText(self.image, f'{label}', (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                            thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)