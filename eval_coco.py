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


def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def main(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_transforms = get_augmentation(config.augmentations, types = 'val')
    retransforms = Compose([
        Denormalize(mean=config.augmentations['mean'], std=config.augmentations['std'], box_transform=False),
        ToPILImage(),
        Resize(size = config.augmentations['image_size'])])

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
    
    ann_path = os.path.join('datasets', config.project_name, config.val_anns)
    img_path = os.path.join('datasets', config.project_name, config.val_imgs)
    coco_gt = COCO(ann_path)
    image_ids = coco_gt.getImgIds()[:args.max_images]

    results = []

    with torch.no_grad():
        batch_size = 2
        with tqdm(total=len(image_ids)) as pbar:
            empty_imgs = 0
            batch = []
            for img_idx, image_id in enumerate(image_ids):
                image_info = coco_gt.loadImgs(image_id)[0]
                image_path = os.path.join(img_path,image_info['file_name'])

                img = Image.open(image_path).convert('RGB')
                
                outputs = []
                
                batch.append(img)
                if ((img_idx+1) % batch_size == 0) or img_idx==len(image_ids)-1:
                    inputs = torch.stack([val_transforms(img)['img'] for img in batch])
                    batch = {'imgs': inputs.to(device)}
                    preds = model.inference_step(batch, args.min_conf, args.min_iou)
                    preds = postprocessing(preds, batch['imgs'].cpu()[0], retransforms, out_format='xywh')
                    outputs += preds
                    batch = []

                try:
                    bbox_xywh = np.concatenate([i['bboxes'] for i in outputs if len(i['bboxes'])>0]) 
                    cls_ids = np.concatenate([i['classes'] for i in outputs if len(i['bboxes'])>0])    
                    cls_conf = np.concatenate([i['scores'] for i in outputs if len(i['bboxes'])>0])
         
                    bbox_xywh, cls_conf, cls_ids = box_nms_numpy(bbox_xywh, cls_conf, cls_ids, threshold=0.01, box_format='xywh')
                except:
                    bbox_xywh = None
                    
                if bbox_xywh is None:
                    empty_imgs += 1
                else:
                    for i in range(bbox_xywh.shape[0]):
                        score = float(cls_conf[i])
                        label = int(cls_ids[i])
                        box = bbox_xywh[i, :]
                        image_result = {
                            'image_id': image_id,
                            'category_id': label + 1,
                            'score': float(score),
                            'bbox': box.tolist(),
                        }

                        results.append(image_result)

                pbar.update(1)
                pbar.set_description(f'Empty images: {empty_imgs}')    

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    filepath = f'results/{config.project_name}_submission.json'
    if not os.path.exists('results'):
        os.mkdir('results')

    if os.path.exists(filepath):
        os.remove(filepath)

    json.dump(results, open(filepath, 'w'), indent=4)

    _eval(coco_gt, image_ids, filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training EfficientDet')
    parser.add_argument('--config' , type=str, help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('--max_images' , type=int, help='max number of images', default=10000)
    parser.add_argument('--weight' , type=str, help='project file that contains parameters')
    parser.add_argument('--min_conf', type=float, default= 0.15, help='minimum confidence for an object to be detect')
    parser.add_argument('--min_iou', type=float, default = 0.3, help='minimum iou threshold for non max suppression')

    args = parser.parse_args()
    config = Config(os.path.join('configs',args.config+'.yaml'))
    main(args, config)