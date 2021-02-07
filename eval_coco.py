from utils.getter import *

import json
import os

import argparse
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models.backbone import EfficientDetBackbone
from utils.utils import postprocessing, box_nms_numpy
from metrics import mAP

def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.params.iouType = "bbox"
    coco_eval.params.iouThrs = np.array([0.4])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def main(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    val_transforms = get_augmentation(config, _type = 'val')

    testset = CocoDataset(
        config = config,
        root_dir=os.path.join('datasets', config.project_name, config.val_imgs), 
        ann_path = os.path.join('datasets', config.project_name, config.val_anns),
        inference = True,
        train = False,
        transforms=val_transforms)

    metric = mAPScores(
        dataset=testset,
        max_images = args.max_images,
        min_conf = args.min_conf,
        min_iou = args.min_iou,
        retransforms = None)

    NUM_CLASSES = len(config.obj_list)
    net = EfficientDetBackbone(num_classes=NUM_CLASSES, compound_coef=args.compound_coef,
                                 ratios=eval(config.anchors_ratios), scales=eval(config.anchors_scales))
    model = Detector(
                    n_classes=NUM_CLASSES,
                    model = net,
                    criterion= FocalLoss(), 
                    optimizer= torch.optim.Adam,
                    optim_params = {'lr': 0.1},     
                    device = device)
    model.eval()

    if args.weight is not None:                
        load_checkpoint(net, args.weight)
    
    metric.update(model)
    metric.value()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training EfficientDet')
    parser.add_argument('--config' , type=str, help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('--max_images' , type=int, help='max number of images', default=10000)
    parser.add_argument('--weight' , type=str, help='project file that contains parameters')
    parser.add_argument('--min_conf', type=float, default= 0.2, help='minimum confidence for an object to be detect')
    parser.add_argument('--min_iou', type=float, default = 0.15, help='minimum iou threshold for non max suppression')

    args = parser.parse_args()
    config = Config(os.path.join('configs',args.config+'.yaml'))
    main(args, config)