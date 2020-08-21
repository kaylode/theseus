from utils.getter import *
import torch.utils.data as data
import torch
import torchvision.models as models
from tqdm import tqdm
import torch.nn as nn
from models.resnet import ResNet34
from torch.optim.lr_scheduler import StepLR

transforms = Compose([
    Resize((300,300)),
    ToTensor(),
    Normalize(),
])

if __name__ == "__main__":
    #logger = Logger()
    
    data_path = "datasets/datasets/Garbage Classification"
    voc_path = "datasets/datasets/VOC/images"
    voc_anno = "datasets/datasets/VOC/annotations/pascal_train2012.json"
    trainset = ImageClassificationDataset(data_path+ "/train", transforms= transforms, shuffle=True)
    valset = ImageClassificationDataset(data_path+ "/val", transforms= transforms, shuffle=True)
    #trainset = ObjectDetectionDataset(img_dir=voc_path, ann_path = voc_anno,transforms= transforms)
    print(trainset)
    print(valset)

    NUM_CLASSES = len(trainset.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    # Dataloader
    BATCH_SIZE = 4
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=trainset.collate_fn, shuffle=True)
    valloader = data.DataLoader(valset, batch_size=BATCH_SIZE, collate_fn=valset.collate_fn, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam
    metrics = [AccuracyMetric(decimals=3)]

    model = ResNet34(NUM_CLASSES,
                     lr = 1e-4,
                     criterion= criterion, 
                     optimizer= optimizer,
                     metrics=  metrics,
                     device = device)
    #load_checkpoint(model, "weights/ResNet34-9.pth")

    cp = Checkpoint(save_per_epoch=1)
    scheduler = StepLR(model.optimizer, step_size=2, gamma=0.1)
    trainer = Trainer(model,
                     trainloader, 
                     valloader,
                     checkpoint = cp,
                     scheduler = scheduler,
                     evaluate_per_epoch = 1)
    
    #results = trainer.inference_batch(valloader)
    #print(valset.classes[results[0]])
    #valset.visualize_item(0)
    
    trainer.fit(num_epochs=10, print_per_iter=50)
    

  
