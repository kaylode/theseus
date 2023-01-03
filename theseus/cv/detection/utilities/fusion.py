import numpy as np
from ensemble_boxes import nms, weighted_boxes_fusion


def box_fusion(
    bounding_boxes,
    confidence_score,
    labels,
    mode="wbf",
    image_size=None,
    weights=None,
    iou_threshold=0.5,
):
    """
    bounding boxes:
        list of boxes of same image [[box1, box2,...],[...]] if ensemble many models
        list of boxes of single image [[box1, box2,...]] if done on one model
        image size: [w,h]
    """

    if image_size is not None:
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        normalized_boxes = []
        for ens_boxes in bounding_boxes:
            if isinstance(ens_boxes, list):
                ens_boxes = np.array(ens_boxes)
            ens_boxes[:, 0] = ens_boxes[:, 0] * 1.0 / image_size[0]
            ens_boxes[:, 1] = ens_boxes[:, 1] * 1.0 / image_size[1]
            ens_boxes[:, 2] = ens_boxes[:, 2] * 1.0 / image_size[0]
            ens_boxes[:, 3] = ens_boxes[:, 3] * 1.0 / image_size[1]
            normalized_boxes.append(ens_boxes)
        normalized_boxes = np.array(normalized_boxes)
    else:
        normalized_boxes = bounding_boxes.copy()

    if mode == "wbf":
        picked_boxes, picked_score, picked_classes = weighted_boxes_fusion(
            normalized_boxes,
            confidence_score,
            labels,
            weights=weights,
            iou_thr=iou_threshold,
            conf_type="avg",  # [nms|avf]
            skip_box_thr=0.0001,
        )
    elif mode == "nms":
        picked_boxes, picked_score, picked_classes = nms(
            normalized_boxes,
            confidence_score,
            labels,
            weights=weights,
            iou_thr=iou_threshold,
        )

    if image_size is not None:
        result_boxes = []
        for ens_boxes in picked_boxes:
            ens_boxes[0] = ens_boxes[0] * image_size[0]
            ens_boxes[1] = ens_boxes[1] * image_size[1]
            ens_boxes[2] = ens_boxes[2] * image_size[0]
            ens_boxes[3] = ens_boxes[3] * image_size[1]
            result_boxes.append(ens_boxes)

    return (
        np.array(result_boxes),
        np.array(picked_score),
        np.array(picked_classes),
    )
