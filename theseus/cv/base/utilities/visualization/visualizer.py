import random
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

from .colors import color_list
from .utils import (
    draw_mask,
    draw_polylines,
    draw_text,
    draw_text_cv2,
    get_font_size,
    reduce_opacity,
)


class Visualizer:
    r"""Visualizer class that do all the visualization stuffs"""

    def __init__(self):
        self.image: Optional[np.ndarray] = None
        self.class_names = None
        self.set_color(color_list)

    def set_color(self, color_list):
        self.color_list = color_list

    def set_image(self, image: np.ndarray) -> None:
        """
        Set the current image
        """
        self.image = image.copy()

        if self.image.dtype == "uint8":
            self.image = self.image / 255.0

    def set_classnames(self, class_names: List[str]) -> None:
        self.class_names = class_names

    def get_image(self) -> np.ndarray:
        """
        Get the current image
        """
        if self.image.dtype == "uint8":
            self.image = np.clip(self.image, 0, 255)
        elif self.image.dtype == "float32" or self.image.dtype == "float64":
            self.image = np.clip(self.image, 0.0, 1.0)
            self.image = (self.image * 255).astype(np.uint8)

        return self.image

    def save_image(self, path: str) -> None:
        """
        Save the image
        """
        cv2.imwrite(path, self.get_image())

    def draw_label(
        self,
        label: str,
        font: Any = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale: int = 2,
        fontColor: Tuple = (0, 0, 1),
        thickness: int = 3,
        outline: Tuple = (0, 0, 0),
        offset: int = 50,
    ):

        """
        Draw text on the image then return

        font: Any
            cv2 font style
        fontScale: int
            size of text
        fontColor: Tuple
            color of text
        thickness: int
            text thickness
        outline:   Tuple
            outline the text, leave None to have disable
        offset:    `int`
            offset to position of text from the bottom
        """

        assert self.image is not None
        if self.class_names is not None:
            label = self.class_names[label]

        h, w, c = self.image.shape
        white_canvas = np.ones((h + offset, w, c))
        white_canvas[:h, :w, :c] = self.image
        bottomLeftCornerOfText = (int(w / 6), h + 10)

        draw_text_cv2(
            white_canvas,
            str(label),
            bottomLeftCornerOfText,
            fontFace=font,
            fontScale=fontScale,
            color=fontColor,
            outline_color=outline,
            thickness=thickness,
        )

        self.image = white_canvas.copy()

    def draw_polygon_ocr(self, polygons, texts=None, font="assets/fonts/aachenb.ttf"):
        image = self.image.copy()
        maskIm = Image.new("L", (self.image.shape[1], self.image.shape[0]), 0)

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
                color = tuple([random.randint(0, 255) / 255.0 for _ in range(3)])
                white_img = draw_text(white_img, text, polygon, font, color, font_size)

        # Mask out polygons
        mask = np.stack([maskIm, maskIm, maskIm], axis=2)
        masked = image * mask

        # Reduce opacity of original image
        o_image = reduce_opacity(image)
        i_masked = (np.bitwise_not(mask) / 255).astype(np.int)
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

    def draw_bbox(self, boxes, labels=None, scores=None) -> None:
        assert self.image is not None

        tl = int(round(0.001 * max(self.image.shape[:2])))  # line thickness

        tup = zip(boxes, labels) if labels is not None else boxes
        tup = zip(boxes, labels, scores) if scores is not None else tup

        for item in tup:
            if labels is not None:
                if scores is not None:
                    box, label, score = item
                else:
                    box, label = item
                    score = None
                color = self.color_list[int(label)]
            else:
                box, label, score = item, None, None
                color = self.color_list[1]

            coord = [box[0], box[1], box[2], box[3]]
            c1, c2 = (int(coord[0]), int(coord[1])), (
                int(coord[2]),
                int(coord[3]),
            )
            cv2.rectangle(self.image, c1, c2, color, thickness=tl * 2)

            if label is not None:
                if self.class_names is not None:
                    label = self.class_names[label]

                if score is not None:
                    score = np.round(score, 3)
                    label = f"{label}: {score}"

                tf = max(tl - 2, 1)  # font thickness
                s_size = cv2.getTextSize(
                    f"{label}", 0, fontScale=float(tl) / 3, thickness=tf
                )[0]
                c2 = c1[0] + s_size[0] + 15, c1[1] - s_size[1] - 3
                cv2.rectangle(self.image, c1, c2, color, -1)  # filled
                cv2.putText(
                    self.image,
                    f"{label}",
                    (c1[0], c1[1] + 2),
                    0,
                    float(tl) / 3,
                    [0, 0, 0],
                    thickness=tf,
                    lineType=cv2.FONT_HERSHEY_SIMPLEX,
                )

    def _tensor_to_numpy(self, image: torch.Tensor) -> np.ndarray:
        """
        Convert torch image to numpy image (C, H, W) --> (H, W, C)
        """
        return image.numpy().squeeze().transpose((1, 2, 0))

    def make_grid(
        self,
        batch: List[torch.Tensor],
        nrow: Optional[int] = None,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Make grid from batch of image
            batch: `List[torch.Tensor]`
                batch of tensor image with shape (C,H,W)
            nrow: `Optional[int]`
                width size of grid
            normalize: `bool`
                whether to normalize the grid in range [0,1]
            return: `torch.Tensor`
                grid image with shape [H*ncol, W*nrow, C]
        """

        if nrow is None:
            nrow = int(np.ceil(np.sqrt(len(batch))))

        batch = torch.stack(batch, dim=0)  # (N, C, H, W)
        grid_img = torchvision.utils.make_grid(batch, nrow=nrow, normalize=normalize)

        return grid_img.permute(1, 2, 0)

    def denormalize(
        self,
        image: Union[torch.Tensor, np.ndarray],
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
    ) -> np.ndarray:
        """
        Denormalize an image and return
        image: `torch.Tensor` or `np.ndarray`
            image to be denormalized
        """
        mean = np.array(mean)
        std = np.array(std)

        if isinstance(image, torch.Tensor):
            img_show = self._tensor_to_numpy(image)
        else:
            img_show = image.copy()

        img_show = img_show * std + mean
        img_show = np.clip(img_show, 0, 1)
        return img_show

    def denormalize_bboxes(
        self, boxes, order=None, image_shape=None, auto_scale: bool = True
    ) -> np.ndarray:
        """
        Denormalize bboxes and return
        image: `torch.Tensor` or `np.ndarray`
            image to be denormalized
        """

        if auto_scale:
            bbox_normalized = False
            if isinstance(boxes, np.ndarray) and np.amax(boxes) <= 1.0:
                bbox_normalized = True
            if isinstance(boxes, torch.Tensor) and torch.max(boxes) <= 1.0:
                bbox_normalized = True
            if bbox_normalized and image_shape is not None:
                boxes[:, [0, 2]] *= image_shape[1]
                boxes[:, [1, 3]] *= image_shape[0]

        if order is not None:
            from theseus.cv.detection.augmentations.bbox_transforms import BoxOrder

            denom = BoxOrder(order)
            boxes = denom.apply_to_bboxes(boxes)
            boxes = np.stack([torch.stack(i, dim=0).numpy() for i in boxes]).astype(int)

        return boxes

    def decode_segmap(
        self, segmap: np.ndarray, num_classes: Optional[int] = None
    ) -> np.ndarray:
        """
        Decode an segmentation mask into colored mask based on class indices

        segmap: `np.ndarray`
            1-channel segmentation masks with each pixel represent one class
        num_classes: `int`
            number of class indices that segmentation mask has

        return: `np.ndarray`
            rgb image, with each color represent different class
        """

        if len(segmap.shape) == 3:  # (NC, H, W), need argmax
            segmap = np.argmax(segmap, axis=0)

        if num_classes is None:
            num_classes = int(np.max(segmap)) + 1

        palette = np.array(self.color_list[:num_classes]) * 255
        palette = palette[:, ::-1].astype(np.uint8)

        segmap = segmap.astype(np.uint8)
        rgb = Image.fromarray(segmap, "P")
        rgb.putpalette(palette)

        return np.array(rgb.convert("RGB"))
