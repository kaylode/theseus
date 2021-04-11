from utils.getter import *
import argparse
import os
import cv2
import matplotlib.pyplot as plt 
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.utils import draw_boxes_v2
from utils.postprocess import box_fusion, postprocessing, change_box_order
import pandas as pd
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from augmentations.transforms import get_resize_augmentation
from augmentations.transforms import MEAN, STD

parser = argparse.ArgumentParser(description='Inference AIC Challenge Dataset')
parser.add_argument('--min_conf', type=float, default= 0.1, help='minimum confidence for an object to be detect')
parser.add_argument('--min_iou', type=float, default=0.5, help='minimum iou threshold for non max suppression')
parser.add_argument('--weight', type=str, default = None,help='version of EfficentDet')
parser.add_argument('--image', type=str, help='path to an image to inference')
args = parser.parse_args() 

class Testset():
    def __init__(self, config, input_path, transforms=None):
        self.input_path = input_path # path to image folder or a single image
        self.transforms = transforms
        self.image_size = config.image_size
        self.load_images()

    def load_images(self):
        self.all_image_paths = []   
        if os.path.isdir(self.input_path):  # path to image folder
            paths = sorted(os.listdir(self.input_path))
            for path in paths:
                self.all_image_paths.append(os.path.join(self.input_path, path))
        elif os.path.isfile(self.input_path): # path to single image
            self.all_image_paths.append(self.input_path)

    def __getitem__(self, idx):
        image_path = self.all_image_paths[idx]
        img = cv2.imread(image_path)
        image_w, image_h = self.image_size
        ori_height, ori_width, c = image.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        return {
            'img': img,
            'image_ori_w': ori_width,
            'image_ori_h': ori_height,
            'image_w': image_w,
            'image_h': image_h,
        }

    def collate_fn(self, batch):
        imgs = torch.stack([s['img'] for s in batch])   
        image_ori_ws = [s['image_ori_w'] for s in batch]
        image_ori_hs = [s['image_ori_h'] for s in batch]
        image_ws = [s['image_w'] for s in batch]
        image_hs = [s['image_h'] for s in batch]
        img_scales = torch.tensor([1.0]*len(batch), dtype=torch.float)
        img_sizes = torch.tensor([imgs[0].shape[-2:]]*len(batch), dtype=torch.float)
        return {
            'imgs': imgs,
            'image_ori_ws': image_ori_ws,
            'image_ori_hs': image_ori_hs,
            'image_ws': image_ws,
            'image_hs': image_hs,
            'img_sizes': img_sizes, 
            'img_scales': img_scales
        }

    def __len__(self):
        return len(self.all_image_paths)


def main(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    num_gpus = len(config.gpu_devices.split(','))

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    test_transforms = A.Compose([
        get_resize_augmentation(config.image_size, keep_ratio=config.keep_ratio),
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=1.0, p=1.0),
        ToTensorV2(p=1.0)
    ])

    if config.tta:
        config.tta = TTA(
            min_conf=config.tta_conf_threshold, 
            min_iou=config.tta_iou_threshold, 
            postprocess_mode=config.tta_ensemble_mode)
    else:
        config.tta = None

    testset = Testset(
        config, 
        args.input_path
        transforms=test_transforms)
    testloader = DataLoader(
        testset,
        batch_size=testset.batch_size,
        
    )

    class_mapping = config.obj_list
    net = get_model(args, config, device, num_classes=len(class_mapping))

    model = Detector(model = net, device = device)
    model.eval()
    if args.weight is not None:                
        load_checkpoint(model, args.weight)
    
    


    empty_imgs = 0
    with tqdm(total=len(testloader)) as pbar:
        with torch.no_grad():
            for idx, batch in enumerate(testloader):
                if config.tta is not None:
                    preds = config.tta.make_tta_predictions(model, batch)
                else:
                    preds = model.inference_step(batch)

                for idx, outputs in enumerate(preds):
                    img_id = batch['img_ids'][idx]
                    ori_img = batch['ori_imgs'][idx]
                    img_w = batch['image_ws'][idx]
                    img_h = batch['image_hs'][idx]
                    img_ori_ws = batch['image_ori_ws'][idx]
                    img_ori_hs = batch['image_ori_hs'][idx]
                    
                    outputs = postprocessing(
                        outputs, 
                        current_img_size=[img_w, img_h],
                        ori_img_size=[img_w, img_h],
                        min_iou=config.min_iou_val,
                        min_conf=config.min_conf_val,
                        max_dets=connfig.max_post_nms,
                        mode=config.fusion_mode)

                    boxes = outputs['bboxes'] 
                    labels = outputs['classes']  
                    scores = outputs['scores']

                    if USE_FILTER:
                        boxes, scores, labels = binary_filter(
                            test_df, 
                            image_id=img_id, 
                            boxes=boxes, 
                            scores=scores, 
                            labels=labels)
                        

                    if len(boxes) == 0:
                        empty_imgs += 1
                        boxes = None

                    if boxes is not None:
                        if args.output_path is not None:
                            out_path = os.path.join(args.output_path, f'{img_id}.png')
                            draw_boxes_v2(out_path, ori_img , boxes, labels, scores, idx_classes)

                    if args.submission:
                        if boxes is not None:
                            for box, score, cls_id in zip(boxes, scores, labels):
                                x,y,w,h = box
                                cls_id = int(cls_id) - 1
                                if config.keep_ratio:
                                    # Subtract left padding of image
                                    if img_w > img_h:
                                        y -= (img_w-img_h)/2 
                                    else:
                                        x -= (img_h-img_w)/2

                                if cls_id != 14:
                                    x = float(x*1.0/img_w)
                                    y = float(y*1.0/img_h)
                                    w = float(w*1.0/img_w)
                                    h = float(h*1.0/img_h)

                                score = np.round(float(score),3)
                                results.append([img_id, cls_id, score, x, y, x+w, y+h])
                        else:
                            results.append([img_id, 14, 1.0, 0, 0, 1, 1])

                
                        
                pbar.update(1)
                pbar.set_description(f'Empty images: {empty_imgs}')

        if args.submission:
            submission_df = pd.DataFrame(results, columns=['image_id', 'class_id', 'score', 'x_min', 'y_min' , 'x_max', 'y_max'])
            submission_df.to_csv('results/folds/test_pred_f4.csv', index=False)


if __name__ == '__main__':
    
    config = Config('./configs/configs.yaml')                   
    main(args, config)
    