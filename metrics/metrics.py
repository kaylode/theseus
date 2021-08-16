import os
import torch
import json
import numpy as np
from tqdm import tqdm
from .metrictemplate import TemplateMetric
from pycocotools.coco import COCO
from .pycocoevalcap.eval import COCOEvalCap

"""
https://github.com/salaniz/pycocoevalcap
"""

"""
GT format
annotation{
  "id": int, 
  "image_id": int, 
  "caption": str,
}

Result format
[{
    "image_id": int, 
    "caption": str,
}]
"""

def _eval(gt_json_path, pred_json_path, image_ids=None):

    coco_gt = COCO(gt_json_path)
    
    if image_ids is None:
        image_ids = coco_gt.getImgIds()

    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    coco_eval = COCOEvalCap(coco_gt, coco_pred)
    coco_eval.params['image_id'] = image_ids

    coco_eval.evaluate()

    # create output dictionary
    stats = {}
    for metric, score in coco_eval.eval.items():
        stats[metric] = score

    return stats


class NLPEval(TemplateMetric):
    def __init__(
            self,
            dataloader, 
            max_samples = 10000,
            decimals = 4):

        self.dataloader = dataloader
        self.max_samples = max_samples
        self.decimals = decimals
        self.filepath = f'results/text_results.json'
        self.gt_filepath = f'results/text_gt.json'
        self.image_ids = []
        self.reset()

        if not os.path.exists('results'):
            os.mkdir('results')
            
    def reset(self):
        self.model = None
        self.image_ids = []

    def update(self, model):
        self.model = model
        self.model.eval()

    def compute(self):
        gt_dict = {
            'images': [],
            'annotations': []
        }
        result_dict = []

        image_id = 0
        with torch.no_grad():
            self.dataloader.create_batches()
            total_iter = min(len(self.dataloader)-1, int(self.max_samples/self.dataloader.batch_size))
            with tqdm(total=total_iter) as pbar:
                for idx, raw_batch in enumerate(self.dataloader.batches):
                    if idx > total_iter:
                        break

                    raw_targets = [s['tgt_text'] for s in raw_batch]
                    batch = self.dataloader.collate_fn(raw_batch)
                    preds = self.model.inference_step(batch, self.dataloader.tgt_tokenizer)

                    for raw_target, pred in zip(raw_targets, preds):

                        gt_dict["images"].append({
                            'id': image_id
                        })

                        gt_dict['annotations'].append({
                            'id': image_id,
                            'image_id': image_id,
                            'caption': raw_target
                        })
                            
                        result_dict.append({
                            'image_id': image_id,
                            'caption': pred
                        })

                        image_id += 1
                    pbar.update(1)

        if not len(result_dict):
            return False

        # write output
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        json.dump(result_dict, open(self.filepath, 'w'), indent=4)

        # Write gt
        if os.path.exists(self.gt_filepath):
            os.remove(self.gt_filepath)
        json.dump(gt_dict, open(self.gt_filepath, 'w'), indent=4)
        
        return True

    def value(self):
        self.compute()
        stats = _eval(self.gt_filepath, self.filepath)
        print(stats)
        return stats

    def __str__(self):
        return f'Mean Average Precision: {self.value()}'

    def __len__(self):
        return len(self.dataloader)

