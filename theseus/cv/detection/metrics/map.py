import os
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

    def __init__(self, image_dir, gt_dir):
        raise NotImplementedError
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.gt_json = "./temp/gt.json"
        self.pred_json = "./temp/pred.json"

        # Convert csv to json
        self.make_gt_json_file()
        self.coco_gt = COCO(self.gt_json)
        self.get_image_ids()

    def make_gt_json_file(self):
        os.makedirs(os.path.dirname(self.gt_json), exist_ok=True)

        my_dict = {
            "images": [],
            "annotations": [],
            "categories": [],
        }

        img_count = 0
        item_count = 0
        self.image_dict = {}
        labels = LABELS

        for label_idx, label in enumerate(labels):
            class_dict = {
                "supercategory": None,
                "id": label_idx + 1,  # Coco starts from 1
                "name": label,
            }
            my_dict["categories"].append(class_dict)


        gt_names = os.listdir(self.gt_dir)

        for gt_name in gt_names: 
            gt_filepath = os.path.join(self.gt_dir, gt_name)

            image_id, _ = os.path.splitext(gt_name)
            image_name = image_id + '.jpg'
            image_path = os.path.join(self.image_dir, image_name)

            img = Image.open(image_path)
            width, height = img.width, img.height

            with open(gt_filepath, 'r') as f:
                annotations = f.read().splitlines()

            for row in annotations:
                item = row.split()
                class_id = int(item[0])
                w = float(item[3]) * width
                h = float(item[4]) * height
                xmin = float(item[1]) * width - w/2
                ymin = float(item[2]) * height - h/2
                xmax = xmin + w
                ymax = ymin + h

                if LIMIT_DICT:
                    limit_w = LIMIT_DICT[class_id]['limit_w']
                    limit_h = LIMIT_DICT[class_id]['limit_h']
                    if w < limit_w[0] or w > limit_w[1] or h < limit_h[0] or h > limit_h[1]:
                        continue

                if image_name not in self.image_dict.keys():
                    self.image_dict[image_name] = img_count
                    img_count += 1
                    image_id = self.image_dict[image_name]
                    img_dict = {
                        "file_name": image_name,
                        "height": height,
                        "width": width,
                        "id": image_id,
                    }
                    my_dict["images"].append(img_dict)

                ann_w = xmax - xmin
                ann_h = ymax - ymin
                image_id = self.image_dict[image_name]
                ann_dict = {
                    "id": item_count,
                    "image_id": image_id,
                    "bbox": [int(xmin), int(ymin), int(ann_w), int(ann_h)],
                    "area": ann_w * ann_h,
                    "category_id": int(class_id) + 1,  # Coco starts from 1
                    "iscrowd": 0,
                }
                item_count += 1
                my_dict["annotations"].append(ann_dict)

        if os.path.isfile(self.gt_json):
            os.remove(self.gt_json)
        with open(self.gt_json, "w") as outfile:
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

        pred_names = os.listdir(path)
        results = []
        for pred_name in pred_names: 
            pred_filepath = os.path.join(path, pred_name)

            image_id, _ = os.path.splitext(pred_name)
            image_name = image_id + '.jpg'
            image_path = os.path.join(self.image_dir, image_name)

            img = Image.open(image_path)
            width, height = img.width, img.height

            with open(pred_filepath, 'r') as f:
                annotations = f.read().splitlines()

            if image_name not in self.image_dict.keys():
                continue

            for row in annotations:
                item = row.split()
                class_id = int(item[0])
                score = float(item[5])
                w = float(item[3]) * width
                h = float(item[4]) * height
                xmin = float(item[1]) * width - w/2
                ymin = float(item[2]) * height - h/2

                if LIMIT_DICT:
                    limit_w = LIMIT_DICT[class_id]['limit_w']
                    limit_h = LIMIT_DICT[class_id]['limit_h']
                    if w < limit_w[0] or w > limit_w[1] or h < limit_h[0] or h > limit_h[1]:
                        continue

                results.append(
                    {
                        "image_id": int(self.image_dict[image_name]),
                        "category_id": class_id + 1,
                        "bbox": [int(xmin), int(ymin), int(w), int(h)],
                        "score": float(score),
                    }
                )

        if os.path.isfile(self.pred_json):
            os.remove(self.pred_json)
        with open(self.pred_json, "w") as outfile:
            json.dump(results, outfile)

    def get_image_ids(self):
        self.image_ids = list(self.image_dict.values())

    def update_pred(self, pred_dir):
        self.make_pred_json_file(pred_dir)

    def value(self):
        # load results in COCO evaluation tool
        coco_pred = self.coco_gt.loadRes(self.pred_json)

        # run COCO evaluation
        coco_eval = COCOeval(self.coco_gt, coco_pred, "bbox")
        coco_eval.params.imgIds = self.image_ids

        coco_eval.params.iouThrs = np.array([MIN_IOU])

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
        # coco_eval.summarize()
        
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

        recalls.insert(0, recall_all)
        precisions.insert(0, precision_all)
        LABELS.insert(0, 'All')

        val_summary = {
            "Object name": LABELS,
            "Precision": precisions,
            "Recall": recalls
        }
        table = tabulate(
            val_summary, headers="keys", tablefmt="fancy_grid"
        )

        print(table)
        pd.DataFrame(val_summary).to_csv(os.path.join(OUTDIR, 'val_summary.csv'), index=False)

        stats = coco_eval.stats
        return stats


if __name__ == "__main__":

    metric = mAPScore(args.image_dir, args.gt_dir)
    metric.update_pred(args.pred_dir)
    metric.evaluate()