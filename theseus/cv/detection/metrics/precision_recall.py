from typing import List, Dict
import os
import numpy as np
import pandas as pd
from tabulate import tabulate

from theseus.base.metrics.metric_template import Metric
from .misc import MatchingPairs, BoxWithLabel

class DetectionPrecisionRecall(Metric):
    def __init__(self, num_classes, min_conf=0.2, min_iou=0.5, eps=1e-6, **kwargs):
        self.eps = eps
        self.min_iou = min_iou
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.all_gt_instances = []
        self.all_pred_instances = []
        self.idx = 0

    def update(self, output, batch):
        target = batch["targets"] 

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

    def value(self):
        total_tp, total_fp, total_fn = self.calculate_cfm(
            self.all_pred_instances, 
            self.all_gt_instances
        )
        score = self.calculate_pr(total_tp, total_fp, total_fn)
        return score

    def calculate_cfm(self, pred_boxes: List[BoxWithLabel], gt_boxes: List[BoxWithLabel]):
        total_fp = []
        total_fn = []
        total_tp = []
        for pred_box, gt_box in zip(pred_boxes, gt_boxes):
            matched_pairs = MatchingPairs(
                pred_box, 
                gt_box, 
                min_iou=self.min_iou, 
                eps=self.eps
            )
            tp = matched_pairs.get_acc()
            fp = matched_pairs.get_false_positive()
            fn = matched_pairs.get_false_negative()
            total_tp += tp
            total_fp += fp
            total_fn += fn
        return total_tp, total_fp, total_fn

    def calculate_pr(self, total_tp, total_fp, total_fn):
        tp_per_class = {
            i: 0 for i in range(self.num_classes)
        }

        fp_per_class = {
            i: 0 for i in range(self.num_classes)
        }

        fn_per_class = {
            i: 0 for i in range(self.num_classes)
        }

        for box, _, _ in total_tp:
            label = box.get_label()
            tp_per_class[label] += 1

        for box in total_fp:
            label = box.get_label()
            fp_per_class[label] += 1

        for box in total_fn:
            label = box.get_label()
            fn_per_class[label] += 1

        precisions = []
        recalls = []
        for cls_id in range(self.num_classes):

            if tp_per_class[cls_id] + fp_per_class[cls_id] == 0:
                precisions.append(-1)
            else:
                precisions.append(
                    tp_per_class[cls_id] / (tp_per_class[cls_id] + fp_per_class[cls_id])
                )

            if tp_per_class[cls_id] + fn_per_class[cls_id] == 0:
                recalls.append(-1)
            else:
                recalls.append(
                    tp_per_class[cls_id] / (tp_per_class[cls_id] + fn_per_class[cls_id])
                )

        np_precisions = np.array(precisions)
        np_recalls = np.array(recalls)

        precision_all = sum(np_precisions[np_precisions!=-1]) / (self.num_classes - sum(np_precisions==-1))
        recall_all = sum(np_recalls[np_recalls!=-1]) / (self.num_classes - sum(np_recalls==-1))

        return {f"precision": precision_all, f"recall": recall_all}
        