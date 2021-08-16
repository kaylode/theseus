import os
import torch
import json
import numpy as np
from tqdm import tqdm
from .metrictemplate import TemplateMetric
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


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

def _eval(coco_gt, image_ids, pred_json_path, **kwargs):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    coco_eval = COCOEvalCap(coco_gt, coco_pred)
    coco_eval.params.imgIds = image_ids

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

        self.coco_gt = COCO(dataloader.dataset.ann_path)
        self.dataloader = dataloader
        self.max_samples = max_samples
        self.decimals = decimals
        self.filepath = f'results/text_results.json'
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
        results = []
        with torch.no_grad():

            with tqdm(total=min(len(self.dataloader), self.max_samples)) as pbar:
                for idx, batch in enumerate(self.dataloader):
                    if idx > self.max_samples:
                        break
                    
                    preds = self.model.inference_step(batch, self.dataloader.tgt_tokenizer)

                    results += preds
                    pbar.update(1)

        if not len(results):
            return False

        # write output
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        json.dump(results, open(self.filepath, 'w'), indent=4)
        return True

    def value(self):
        result = self.compute()
        valid_imgs = self.coco_gt.getImgIds()

        stats = _eval(self.coco_gt, valid_imgs, self.filepath)
        
        return stats

    def __str__(self):
        return f'Mean Average Precision: {self.value()}'

    def __len__(self):
        return len(self.dataloader)

