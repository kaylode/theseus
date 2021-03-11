from utils.getter import *
import argparse
import os
import cv2
import matplotlib.pyplot as plt 
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.utils import draw_boxes_v2, change_box_order
from utils.postprocess import box_fusion, postprocessing
import pandas as pd
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from augmentations.transforms import get_resize_augmentation

BATCH_SIZE = 16

class TestDataset(Dataset):
    def __init__(self, config, test_df, transforms=None):
        self.image_size = config.image_size
        self.root_dir = os.path.join('datasets', config.project_name, config.train_imgs)
        self.test_df = test_df
        self.transforms = transforms
        self.resize_transforms = get_resize_augmentation(config.image_size, config.keep_ratio, box_transforms=False)
        self.load_data()

    def load_data(self):
        self.fns = [
            annotations for annotations in zip(
                self.test_df['image_id'], self.test_df['width'], self.test_df['height']
            )
        ]

    def __getitem__(self, idx):
        image_id, image_ori_w, image_ori_h = self.fns[idx]
        img_path = os.path.join(self.root_dir, image_id+'.png')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        label = np.array(0)
        
        image_h, image_w, c = img.shape

        if self.resize_transforms is not None:
            resized = self.resize_transforms(image=img, class_labels=label)
            img_ = resized['image']
            

        if self.transforms is not None:
            img = self.transforms(image=img_)['image']
        return {
            'ori_image': img_,
            'image_id': image_id,
            'img': img,
            'image_ori_w': image_ori_w,
            'image_ori_h': image_ori_h,
            'image_w': image_w,
            'image_h': image_h,
        }

    def collate_fn(self, batch):
        imgs = torch.stack([s['img'] for s in batch])   
        img_ids = [s['image_id'] for s in batch]
        ori_imgs = [s['ori_image'] for s in batch]
        image_ori_ws = [s['image_ori_w'] for s in batch]
        image_ori_hs = [s['image_ori_h'] for s in batch]
        image_ws = [s['image_w'] for s in batch]
        image_hs = [s['image_h'] for s in batch]
        img_scales = torch.tensor([1.0]*len(batch), dtype=torch.float)
        img_sizes = torch.tensor([imgs[0].shape[-2:]]*len(batch), dtype=torch.float)
        return {
            'imgs': imgs,
            'ori_imgs': ori_imgs,
            'img_ids': img_ids,
            'image_ori_ws': image_ori_ws,
            'image_ori_hs': image_ori_hs,
            'image_ws': image_ws,
            'image_hs': image_hs,
            'img_sizes': img_sizes, 
            'img_scales': img_scales
        }

    def __len__(self):
      return len(self.fns)

def main(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    num_gpus = len(config.gpu_devices.split(','))

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    if args.output_path is not None:
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

    test_df = pd.read_csv('./datasets/train_info.csv')
    test_transforms = A.Compose([
        A.Resize(
            height = config.image_size[1],
            width = config.image_size[0]),
        ToTensorV2(p=1.0)
    ])

    testset = TestDataset(config, test_df, test_transforms)
    testloader = DataLoader(testset, batch_size = BATCH_SIZE, collate_fn=testset.collate_fn)
    idx_classes = {idx:i for idx,i in enumerate(config.obj_list)}
    NUM_CLASSES = len(config.obj_list)

    if config.tta:
        config.tta = TTA(
            min_conf=config.min_conf_val, 
            min_iou=config.min_iou_val, 
            postprocess_mode=config.tta_ensemble_mode)
    else:
        config.tta = None

    net = get_model(args, config)

    model = Detector(model = net, device = device)
    model.eval()
    if args.weight is not None:                
        load_checkpoint(model, args.weight)
    
    results = []
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
                    # img_ori_ws = batch['image_ori_ws'][idx]
                    # img_ori_hs = batch['image_ori_hs'][idx]
                    
                    outputs = postprocessing(
                        outputs, 
                        current_img_size=[img_w, img_h],
                        ori_img_size=[img_w, img_h],
                        min_iou=config.min_iou_val,
                        min_conf=config.min_conf_val,
                        mode=config.fusion_mode)

                    boxes = outputs['bboxes'] 
                    labels = outputs['classes']  
                    scores = outputs['scores']

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

                                x = float(x*1.0/img_w)
                                y = float(y*1.0/img_h)
                                w = float(w*1.0/img_w)
                                h = float(h*1.0/img_h)
                                score = np.round(float(score),2)
                                results.append([img_id, cls_id, score, x, y, x+w, y+h])
                        else:
                            results.append([img_id, 14, 1.0, 0, 0, 1, 1])

                
                        
                pbar.update(1)
                pbar.set_description(f'Empty images: {empty_imgs}')

        if args.submission:
            submission_df = pd.DataFrame(results, columns=['image_id', 'class_id', 'score', 'x_min', 'y_min' , 'x_max', 'y_max'])
            submission_df.to_csv('results/0_train_pred.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference AIC Challenge Dataset')
    parser.add_argument('config', type=str, default = None,help='save detection at')
    parser.add_argument('--min_conf', type=float, default= 0.001, help='minimum confidence for an object to be detect')
    parser.add_argument('--min_iou', type=float, default=0.5, help='minimum iou threshold for non max suppression')
    parser.add_argument('--weight', type=str, default = 'weights/efficientdet-d2.pth',help='version of EfficentDet')
    parser.add_argument('--output_path', type=str, default = None, help='name of output to .avi file')
    parser.add_argument('--submission', action='store_true', default = False, help='output to submission file')

    args = parser.parse_args() 
    config = Config(os.path.join('configs',args.config+'.yaml'))                   
    main(args, config)
    