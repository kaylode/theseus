from models.detector import Detector
import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from .model import RetinaNet

class RetinaDetector(Detector):
    def __init__(self, n_classes, **kwargs):
        super(RetinaDetector, self).__init__(n_classes = n_classes, **kwargs)
        self.model = RetinaNet(num_classes = n_classes)
        self.model_name = "RetinaNet"
        self.optimizer = self.optimizer(self.parameters(), lr= self.lr)
        self.criterion = self.criterion(n_classes)
        self.n_classes = n_classes
    
        if self.freeze:
            for params in self.model.parameters():
                params.requires_grad = False

        if self.device:
            self.model.to(self.device)
            self.criterion.to(self.device)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        inputs = batch["imgs"]
        boxes = batch['boxes']
        labels = batch['labels']

        if self.device:
            inputs = inputs.to(self.device)
            boxes = boxes.to(self.device)
            labels = labels.to(self.device)
        
        loc_preds, cls_preds = self(inputs)
        loss = self.criterion(loc_preds, cls_preds, boxes, labels)
        return loss

    
    def inference_step(self, batch):
        inputs = batch["imgs"]
        boxes = batch['boxes']
        labels = batch['labels']

        if self.device:
            inputs = inputs.to(self.device)
            boxes = boxes.to(self.device)
            labels = labels.to(self.device)

        loc_preds, cls_preds = self(inputs)
        

        if self.device:
            loc_preds = loc_preds.cpu()
            cls_preds = cls_preds.cpu()
        
        return (loc_preds, cls_preds)

    def evaluate_step(self, batch):
        inputs = batch["imgs"]
        boxes = batch['boxes']
        labels = batch['labels']

        if self.device:
            inputs = inputs.to(self.device)
            boxes = boxes.to(self.device)
            labels = labels.to(self.device)

        loc_preds, cls_preds = self(inputs)
        loss = self.criterion(loc_preds, cls_preds, boxes, labels)
        

        metric_dict = self.update_metrics(
            outputs = {},
            targets={})
        
        return loss , metric_dict

    

    