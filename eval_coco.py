from utils.getter import *

import json
import os

import argparse
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

seed_everything()

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
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    num_gpus = len(config.gpu_devices.split(','))

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    val_transforms = get_augmentation(config, _type = 'val')

    _, _, testset, _, _ = get_dataset_and_dataloader(config)

    if config.tta:
        config.tta = TTA(
            min_conf=config.min_conf_val, 
            min_iou=config.min_iou_val, 
            postprocess_mode=config.tta_ensemble_mode)
    else:
        config.tta = None

    metric = mAPScores(
        dataset=testset,
        min_conf = config.min_conf_val,
        min_iou = config.min_iou_val,
        tta=config.tta,
        max_images=config.max_images_val,
        mode=config.fusion_mode)

    net = get_model(args, config)

    model = Detector(model = net, device = device)
    model.eval()

    if args.weight is not None:                
        load_checkpoint(model, args.weight)
    
    metric.update(model)
    metric.value()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training EfficientDet')
    parser.add_argument('config' , type=str, help='project file that contains parameters')
    parser.add_argument('--max_images' , type=int, help='max number of images', default=10000)
    parser.add_argument('--weight' , type=str, help='project file that contains parameters')

    args = parser.parse_args()
    config = Config(os.path.join('configs',args.config+'.yaml'))
    main(args, config)