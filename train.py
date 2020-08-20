from utils.getter import *
import torch.utils.data as data
import torch
import torchvision.models as models
from tqdm import tqdm
import torch.nn as nn

transforms = Compose([
    Resize((300,300)),
    ToTensor(),
    Normalize(),
])

if __name__ == "__main__":
    logger = Logger()
    logger.log('lo')
    """
    data_path = "datasets/datasets/Garbage Classification"
    voc_path = "datasets/datasets/VOC/images"
    voc_anno = "datasets/datasets/VOC/annotations/pascal_train2012.json"
    trainset = ImageClassificationDataset(data_path+ "/train", transforms= transforms, shuffle=True)
    valset = ImageClassificationDataset(data_path+ "/val", transforms= transforms, shuffle=True)
    #trainset = ObjectDetectionDataset(img_dir=voc_path, ann_path = voc_anno,transforms= transforms)
    print(trainset)
    print(valset)

    NUM_CLASSES = len(trainset.classes)
    print(NUM_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)
    # Dataloader
    BATCH_SIZE = 32
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam
    metrics = [AccuracyMetric(decimals=3)]

    model = ResNet34(NUM_CLASSES,
                     lr = 1e-4,
                     criterion= criterion, 
                     optimizer= optimizer,
                     metrics=  metrics,
                     device = device)
    #load_checkpoint(model, "weights/ResNet34-12.pth")

    cp = Checkpoint(save_per_epoch=6)
    trainer = Trainer(model,
                     trainloader, 
                     valloader,
                     checkpoint = cp, 
                     evaluate_per_epoch = 2)
    
    
    trainer.fit(num_epochs=30)"""
  
