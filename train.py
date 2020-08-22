from utils.getter import *
import torch.utils.data as data
import torch
import torchvision.models as models
from tqdm import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from models.ssd.model import MultiBoxLoss

transforms = Compose([
    Resize((300,300)),
    ToTensor(),
    Normalize(),
])

if __name__ == "__main__":
    
    data_path = "datasets/datasets/Garbage Classification"
    voc_path = "datasets/datasets/VOC/images"
    voc_anno = {
        "train": "datasets/datasets/VOC/annotations/pascal_train2012.json",
        "val": "datasets/datasets/VOC/annotations/pascal_val2012.json"}
    #trainset = ImageClassificationDataset(data_path+ "/train", transforms= transforms, shuffle=True)
    #valset = ImageClassificationDataset(data_path+ "/val", transforms= transforms, shuffle=True)
    trainset = ObjectDetectionDataset(img_dir=voc_path, ann_path = voc_anno['train'],transforms= transforms)
    valset = ObjectDetectionDataset(img_dir=voc_path, ann_path = voc_anno['val'],transforms= transforms)
    print(trainset)
    print(valset)

    NUM_CLASSES = len(trainset.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    # Dataloader
    BATCH_SIZE = 4
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=trainset.collate_fn, shuffle=True)
    valloader = data.DataLoader(valset, batch_size=BATCH_SIZE, collate_fn=valset.collate_fn, shuffle=False)
    
    criterion = MultiBoxLoss
    optimizer = torch.optim.Adam
    metrics = [AccuracyMetric(decimals=3)]
    
    model = Detector(NUM_CLASSES,
                     lr = 1e-3,
                     criterion= criterion, 
                     optimizer= optimizer,
                     metrics=  metrics,
                     device = device)
    
    #load_checkpoint(model, "weights/ResNet34-9.pth")
    model.unfreeze()
    trainer = Trainer(model,
                     trainloader, 
                     valloader,
                     clip_grad = 1.0,
                     checkpoint = Checkpoint(save_per_epoch=1),
                     scheduler = StepLR(model.optimizer, step_size=5, gamma=0.5),
                     evaluate_per_epoch = 15)
    
    print(trainer)
  
    #results = trainer.inference_batch(valloader)
    #print(valset.classes[results[0]])
    #valset.visualize_item(0)
    
    trainer.fit(num_epochs=15, print_per_iter=10)
    

  
