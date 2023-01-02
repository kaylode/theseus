"""
COCO Mean Average Precision Evaluation
True Positive (TP): Predicted as positive as was correct
False Positive (FP): Predicted as positive but was incorrect
False Negative (FN): Failed to predict an object that was there
if IOU prediction >= IOU threshold, prediction is TP
if 0 < IOU prediction < IOU threshold, prediction is FP
Precision measures how accurate your predictions are. Precision = TP/(TP+FP)
Recall measures how well you find all the positives. Recal = TP/(TP+FN)
Average Precision (AP) is finding the area under the precision-recall curve.
Mean Average  Precision (MAP) is AP averaged over all categories.
AP@[.5:.95] corresponds to the average AP for IoU from 0.5 to 0.95 with a step size of 0.05
AP@.75 means the AP with IoU=0.75
*Under the COCO context, there is no difference between AP and mAP
Example execution:
python evaluate.py --gt_csv=0_val.csv --pred_csv=0_predict.csv
"""

import os
import os.path as osp
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
from theseus.base.metrics.metric_template import Metric
from .misc import BoxWithLabel

class MeanAveragePrecision(Metric):
    """
    COCO Mean Average Precision Evaluation
    True Positive (TP): Predicted as positive as was correct
    False Positive (FP): Predicted as positive but was incorrect
    False Negative (FN): Failed to predict an object that was there
    if IOU prediction >= IOU threshold, prediction is TP
    if 0 < IOU prediction < IOU threshold, prediction is FP
    Precision measures how accurate your predictions are. Precision = TP/(TP+FP)
    Recall measures how well you find all the positives. Recal = TP/(TP+FN)
    Average Precision (AP) is finding the area under the precision-recall curve.
    Mean Average  Precision (MAP) is AP averaged over all categories.
    AP@[.5:.95] corresponds to the average AP for IoU from 0.5 to 0.95 with a step size of 0.05
    AP@.75 means the AP with IoU=0.75
    *Under the COCO context, there is no difference between AP and mAP
    """

    def __init__(self, num_classes, classnames, min_iou=0.5, tmp_save_dir='./.cache', **kwargs):
        
        self.num_classes = num_classes
        self.classnames = classnames
        self.min_iou = min_iou
        self.tmp_save_dir = tmp_save_dir
        os.makedirs(self.tmp_save_dir, exist_ok=True)

        self.gt_json = osp.join(self.tmp_save_dir, 'gt_coco.json')
        self.pred_json = osp.join(self.tmp_save_dir, 'pred_coco.json')
        self.reset()

    def reset(self):
        self.all_gt_instances = []
        self.all_pred_instances = []
        self.img_infos = []
        self.img_names = []
        self.image_id_dict = {}
        self.idx = 0

    def update(self, output, batch):
        img_sizes = batch['img_sizes']
        width, height = img_sizes[0, -2:]
        target = batch["targets"] 
        img_ids = batch['img_ids']
        image_names = batch['img_names']
        self.img_infos.extend([{
            'image_id': img_id,
            'image_name': img_name,
            'width': int(width),
            'height': int(height)
        } for img_id, img_name in zip(img_ids, image_names)])

        for pred, gt in zip(output, target):
            pred_boxes = pred['boxes'].cpu().numpy().tolist()
            pred_clss = pred['labels'].cpu().numpy().tolist()
            pred_scores = pred['scores'].cpu().numpy().tolist()

            gt_boxes = gt['boxes'].numpy().tolist()
            gt_clss = gt['labels'].numpy().tolist()

            gt_instances = [
                BoxWithLabel(self.idx, box, int(cls), 1.0) 
                for box, cls in zip(gt_boxes, gt_clss)
            ]
            pred_instances = [
                BoxWithLabel(self.idx, box, int(cls), scr) 
                for (box, cls, scr) in zip(pred_boxes, pred_clss, pred_scores)
            ]
            self.idx+=1
            self.all_gt_instances.append(gt_instances)
            self.all_pred_instances.append(pred_instances)

    def make_gt_json_file(self, path):

        my_dict = {
            "images": [],
            "annotations": [],
            "categories": [],
        }

        img_count = 0
        item_count = 0
        

        for label_idx, label in enumerate(self.classnames):
            class_dict = {
                "supercategory": None,
                "id": label_idx, 
                "name": label,
            }
            my_dict["categories"].append(class_dict)

        for instance_info, instance in zip(self.img_infos, self.all_gt_instances):
            instance_id = instance_info['image_id']
            instance_name = instance_info['image_name']
            height, width = instance_info['height'], instance_info['width']
            if instance_id not in self.image_id_dict.keys():
                self.image_id_dict[instance_id] = img_count
                img_count += 1
                image_id = self.image_id_dict[instance_id]
                img_dict = {
                    "file_name": instance_name,
                    "height": height,
                    "width": width,
                    "id": image_id,
                }
                my_dict["images"].append(img_dict)

            for item in instance:

                class_id = int(item.get_label())
                xmin, ymin, xmax, ymax = item.get_box()

                ann_w = xmax - xmin
                ann_h = ymax - ymin
                image_id = self.image_id_dict[instance_id]
                ann_dict = {
                    "id": item_count,
                    "image_id": image_id,
                    "bbox": [int(xmin), int(ymin), int(ann_w), int(ann_h)],
                    "area": ann_w * ann_h,
                    "category_id": int(class_id),  
                    "iscrowd": 0,
                }
                item_count += 1
                my_dict["annotations"].append(ann_dict)

        if osp.isfile(path):
            os.remove(path)
        with open(path, "w") as outfile:
            json.dump(my_dict, outfile)

    def make_pred_json_file(self, path):
        """
           Output .json format example: (source: https://cocodataset.org/#format-results)
            [{
                "image_id": int, 
                "category_id": int, 
                "bbox": [x,y,width,height], 
                "score": float,
            }]
        """

        results = []

        for instance_info, instance in zip(self.img_infos, self.all_pred_instances):
            instance_id = instance_info['image_id']
            for item in instance:
                class_id = int(item.get_label())
                xmin, ymin, xmax, ymax = item.get_box()
                score = item.get_score()

                results.append(
                    {
                        "image_id": int(self.image_id_dict[instance_id]),
                        "category_id": class_id,
                        "bbox": [int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin)],
                        "score": float(score),
                    }
                )

        if osp.isfile(self.pred_json):
            os.remove(self.pred_json)
        with open(self.pred_json, "w") as outfile:
            json.dump(results, outfile) 

    
    def value(self):
        self.make_gt_json_file(self.gt_json)
        self.make_pred_json_file(self.pred_json)
        coco_gt = COCO(self.gt_json)
        coco_pred = coco_gt.loadRes(self.pred_json)

        # run COCO evaluation
        coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
        coco_eval.params.imgIds = list(self.image_id_dict.keys())
        coco_eval.params.iouThrs = np.array([self.min_iou])

        # Some other params for COCO eval
        # imgIds = []
        # catIds = []
        # iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        # recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        # maxDets = [1, 10, 100]
        # areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        # areaRngLbl = ['all', 'small', 'medium', 'large']
        # useCats = 1

        coco_eval.evaluate()
        coco_eval.accumulate()
        
        recall_stat = coco_eval.eval['recall']
        precision_stat = coco_eval.eval['precision']

        num_classes = recall_stat.shape[1]

        recalls = []
        precisions = []

        for i in range(num_classes):
            recall_class = recall_stat[:, i, 0, -1]
            precision_class = precision_stat[:, :,i, 0, -1]
            
            recall_class = recall_class[recall_class>-1]
            ar = np.mean(recall_class) if recall_class.size else -1

            precision_class = precision_class[precision_class>-1]
            ap = np.mean(precision_class) if precision_class.size else -1

            recalls.append(ar)
            precisions.append(ap)

        np_precisions = np.array(precisions)
        np_recalls = np.array(recalls)



        precision_all = sum(np_precisions[np_precisions!=-1]) / (num_classes - sum(np_precisions==-1))
        recall_all = sum(np_recalls[np_recalls!=-1]) / (num_classes - sum(np_recalls==-1))

        f1_score = 2*precision_all*recall_all / (precision_all + recall_all)
        return {f"precision": precision_all, f"recall": recall_all, 'f1_score': f1_score}