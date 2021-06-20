from utils.getter import *
import argparse
import os
import cv2
import json
import torch
from torch import nn
from torch.utils.data import  DataLoader
import pandas as pd
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from augmentations.transforms import get_resize_augmentation
from augmentations.transforms import MEAN, STD

parser = argparse.ArgumentParser(description='Classify an image / folder of images')
parser.add_argument('--weight', type=str ,help='trained weight')
parser.add_argument('--input_path', type=str, help='path to an image to inference')
parser.add_argument('--output_path', type=str, help='path to save csv result file')

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
            print(paths)
            for path in paths:
                self.all_image_paths.append(os.path.join(self.input_path, path))
        elif os.path.isfile(self.input_path): # path to single image
            self.all_image_paths.append(self.input_path)

    def __getitem__(self, idx):
        image_path = self.all_image_paths[idx]
        image_name = os.path.basename(image_path)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        return {
            'img': img,
            'img_name': image_name,
        }

    def collate_fn(self, batch):
        imgs = torch.stack([s['img'] for s in batch])  
        img_names = [s['img_name'] for s in batch]
        return {
            'imgs': imgs,
            'img_names': img_names,
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

    # if config.tta:
    #     config.tta = TTA(
    #         min_conf=config.tta_conf_threshold, 
    #         min_iou=config.tta_iou_threshold, 
    #         postprocess_mode=config.tta_ensemble_mode)
    # else:
    #     config.tta = None

    testset = Testset(
        config, 
        args.input_path,
        transforms=test_transforms)

    testloader = DataLoader(
        testset,
        batch_size=testset.get_batch_size(),
        num_workers=2,
        pin_memory=True,
        collate_fn=testset.collate_fn
    )

    if args.weight is not None:
        class_names, num_classes = get_class_names(args.weight)


    net = BaseTimmModel(
        name=config.model_name, 
        num_classes=num_classes)

    model = Classifier( model = net,  device = device)

    model.eval()
    if args.weight is not None:                
        load_checkpoint(model, args.weight)

    ## Print info
    print(config)
    print(testset)
    print(f"Nubmer of gpus: {num_gpus}")
    print(devices_info)



    image_names = []
    pred_list = []
    prob_list = []

    with tqdm(total=len(testloader)) as pbar:
        with torch.no_grad():
            for idx, batch in enumerate(testloader):
                
                preds, probs = model.inference_step(batch, return_probs=True)

                for idx, (pred, prob) in enumerate(zip(preds, probs)):
                    img_name = batch['img_names'][idx]
                    image_names.append(img_name)
                    pred_list.append(class_names[pred])
                    prob_list.append(prob)

    result_df = pd.DataFrame({
        'image_names':image_names,
        'predictions': pred_list,
        'probalities': prob_list
    })


    result_df.to_csv(args.output_path, index=False)

    print(f"Result file is saved to {args.output_path}")

if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config(args.weight)
    if config is None:
        print("Config not found. Load configs from configs/configs.yaml")
        config = Config(os.path.join('configs','configs.yaml'))
    else:
        print("Load configs from weight")
    main(args,config)