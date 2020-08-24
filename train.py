from utils.getter import *
import torch.utils.data as data
import torch
import torchvision.models as models
from tqdm import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from models.retinanet.loss import FocalLoss
from models.retinanet.detector import RetinaDetector
from models.retinanet.retina_collator import RetinaNetCollator

from models.ssd.detector import SSDDetector
from models.ssd.model import MultiBoxLoss

transforms = Compose([
    Resize((300,300)),
    ToTensor(),
    Normalize(),
])

if __name__ == "__main__":
    
 
    dataset_path = "datasets/datasets/VOC/"
    img_path = dataset_path + "images"
    ann_path = {
        "train": dataset_path + "annotations/pascal_train2012.json",
        "val": dataset_path + "annotations/pascal_val2012.json"}
   
    trainset = ObjectDetectionDataset(img_dir=img_path, ann_path = ann_path['train'],transforms= transforms)
    valset = ObjectDetectionDataset(img_dir=img_path, ann_path = ann_path['val'],transforms= transforms)
    print(trainset)
    print(valset)

    
    NUM_CLASSES = len(trainset.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    # Dataloader
    BATCH_SIZE = 4
    my_collate = trainset.collate_fn#RetinaNetCollator() #trainset.collate_fn, valset.collate_fn
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=True)
    valloader = data.DataLoader(valset, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=False)
    
    criterion = MultiBoxLoss
    optimizer = torch.optim.SGD
    #metrics = [AccuracyMetric(decimals=3)]
    
    model = SSDDetector(
                    n_classes = NUM_CLASSES,
                    lr = 1e-3,
                    criterion= criterion, 
                    optimizer= optimizer,
 #                   metrics=  metrics,
                    device = device)
    
    #load_checkpoint(model, "weights/RetinaNet-10.pth")
    #model.unfreeze()
    trainer = Trainer(model,
                     trainloader, 
                     valloader,
                     clip_grad = 1.0,
                     checkpoint = Checkpoint(save_per_epoch=5),
                     scheduler = StepLR(model.optimizer, step_size=20, gamma=0.1),
                     evaluate_per_epoch = 50)
    
    print(trainer)
    
    #results = trainer.inference_batch(valloader)
    #print(valset.classes[results[0]])
    #valset.visualize_item(0)
    
    trainer.fit(num_epochs=50, print_per_iter=10)
    

  
