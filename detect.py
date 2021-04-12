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
parser.add_argument('--input_path', type=str, help='path to an image to inference')
parser.add_argument('--output_path', type=str, help='path to save inferenced image')
args = parser.parse_args() 


class_mapping = [
    'background',
]

class Testset():
    def __init__(self, config, input_path, transforms=None):
        self.input_path = input_path # path to image folder or a single image
        self.transforms = transforms
        self.image_size = config.image_size
        self.load_images()

    def get_batch_size(self):
        num_samples = len(self.all_image_paths)

        # Temporary
        return 1

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
        image_name = os.path.basename(image_path)
        img = cv2.imread(image_path)
        image_w, image_h = self.image_size
        ori_height, ori_width, c = image.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        ori_img = img.copy()
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        return {
            'img': img,
            'img_name': image_name,
            'ori_img': ori_img,
            'image_ori_w': ori_width,
            'image_ori_h': ori_height,
            'image_w': image_w,
            'image_h': image_h,
        }

    def collate_fn(self, batch):
        imgs = torch.stack([s['img'] for s in batch])   
        ori_imgs = [s['ori_img'] for s in batch]
        img_names = [s['img_name'] for s in batch]
        image_ori_ws = [s['image_ori_w'] for s in batch]
        image_ori_hs = [s['image_ori_h'] for s in batch]
        image_ws = [s['image_w'] for s in batch]
        image_hs = [s['image_h'] for s in batch]
        img_scales = torch.tensor([1.0]*len(batch), dtype=torch.float)
        img_sizes = torch.tensor([imgs[0].shape[-2:]]*len(batch), dtype=torch.float)
        return {
            'imgs': imgs,
            'ori_imgs': ori_imgs,
            'img_names': img_names,
            'image_ori_ws': image_ori_ws,
            'image_ori_hs': image_ori_hs,
            'image_ws': image_ws,
            'image_hs': image_hs,
            'img_sizes': img_sizes, 
            'img_scales': img_scales
        }

    def __len__(self):
        return len(self.all_image_paths)

    def __str__(self):
        return f"Number of found images: {len(self.all_image_paths)}"

def main(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    num_gpus = len(config.gpu_devices.split(','))
    devices_info = get_devices_info(config.gpu_devices)

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
        args.input_path,
        transforms=test_transforms)
    testloader = DataLoader(
        testset,
        batch_size=testset.get_batch_size(),
        num_workers=2,
        pin_memory=True  
    )

    net = get_model(args, config, device, num_classes=len(class_mapping))

    model = Detector(model = net, device = device)
    model.eval()
    if args.weight is not None:                
        load_checkpoint(model, args.weight)

    if os.path.isdir(args.input_path):
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)


    ## Print info
    print(config)
    print(testset)
    print(f"Nubmer of gpus: {num_gpus}")
    print(devices_info)


    empty_imgs = 0
    with tqdm(total=len(testloader)) as pbar:
        with torch.no_grad():
            for idx, batch in enumerate(testloader):
                if config.tta is not None:
                    preds = config.tta.make_tta_predictions(model, batch)
                else:
                    preds = model.inference_step(batch)

                for idx, outputs in enumerate(preds):
                    img_name = batch['img_names'][idx]
                    ori_img = batch['ori_imgs'][idx]
                    img_w = batch['image_ws'][idx]
                    img_h = batch['image_hs'][idx]
                    img_ori_ws = batch['image_ori_ws'][idx]
                    img_ori_hs = batch['image_ori_hs'][idx]
                    
                    outputs = postprocessing(
                        outputs, 
                        current_img_size=[img_w, img_h],
                        ori_img_size=[img_ori_ws, img_ori_hs],
                        min_iou=config.min_iou_val,
                        min_conf=config.min_conf_val,
                        max_dets=config.max_post_nms,
                        keep_ratio=config.keep_ratio,
                        output_format='xywh',
                        mode=config.fusion_mode)

                    boxes = outputs['bboxes'] 
                    labels = outputs['classes']  
                    scores = outputs['scores']

                    if len(boxes) == 0:
                        empty_imgs += 1
                        boxes = None

                    if boxes is not None:
                        out_path = os.path.join(args.output_path, f'{img_name}.png')
                        draw_boxes_v2(out_path, ori_img , boxes, labels, scores, class_mapping)

                pbar.update(1)
                pbar.set_description(f'Empty images: {empty_imgs}')

if __name__ == '__main__':
    config = Config('./configs/configs.yaml')                   
    main(args, config)
    