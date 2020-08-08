from utils.getter import *
import numpy as np
import random
import torch.utils.data as data
import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms



transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == "__main__":
    trainset = ImageClassificationDataset("datasets/garbage_train", transforms= transforms,max_samples=100)
    valset = ImageClassificationDataset("datasets/garbage_val", transforms= transforms,max_samples=10, shuffle=True)
    print(trainset)
    print(valset)
    
    NUM_CLASSES = len(trainset.classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Dataloader
    BATCH_SIZE = 2
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)
    
    EPOCHS = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam
    metrics = [F1ScoreMetric(NUM_CLASSES, average="macro"), AccuracyMetric(decimals=3)]

    model = ResNet34(NUM_CLASSES,
                     lr = 1e-3,
                     criterion= criterion, 
                     optimizer= optimizer,
                     metrics=  metrics,
                     device = device)

    cp = Checkpoint(save_per_epoch=2)
    trainer = Trainer(model,
                     trainloader, 
                     valloader,
                     checkpoint = cp, 
                     evaluate_per_epoch = 2)
    
    
    trainer.fit(print_per_iter=10)
  
