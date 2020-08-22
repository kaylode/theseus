from .base_model import BaseModel
import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from .ssd.model import SSD300


class Detector(BaseModel):
    def __init__(self, n_classes, **kwargs):
        super(Detector, self).__init__(**kwargs)
        self.model = SSD300(n_classes = n_classes)
        self.model_name = "SSD300"
        self.optimizer = self.optimizer(self.parameters(), lr= self.lr)
        self.criterion = self.criterion(self.model.priors_cxcy)
        self.n_classes = n_classes

        if self.freeze:
            for params in self.model.parameters():
                params.requires_grad = False

        if self.device:
            self.model.to(self.device)
            #self.criterion.to(device)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        inputs = batch["imgs"]
        boxes = batch['boxes']
        labels = batch['labels']

        if self.device:
            inputs = inputs.to(self.device)
            boxes = [x.to(self.device) for x in boxes]
            labels = [x.to(self.device) for x in labels]
        
        loc_preds, cls_preds = self(inputs)
        loss = self.criterion(loc_preds, cls_preds, boxes, labels)
        return loss

    
    def inference_step(self, batch):
        inputs = batch["imgs"]
        boxes = batch['boxes']
        labels = batch['labels']

        if self.device:
            inputs = inputs.to(self.device)
            boxes = [x.to(self.device) for x in boxes]
            labels = [x.to(self.device) for x in labels]

        outputs = self(inputs)
        

        if self.device:
            outputs = [i.cpu().numpy() for i in outputs]
        return outputs

    def evaluate_step(self, batch):
        inputs = batch["imgs"]
        boxes = batch['boxes']
        labels = batch['labels']

        if self.device:
            inputs = inputs.to(self.device)
            boxes = [x.to(self.device) for x in boxes]
            labels = [x.to(self.device) for x in labels]

        loc_preds, cls_preds = self(inputs)
        loss = self.criterion(loc_preds, cls_preds, boxes, labels)
        

        det_boxes, det_labels, det_scores = self.model.detect_objects(
            loc_preds,
            cls_preds,
            min_score=0.01,
            max_overlap=0.45,
            top_k=200)

        metric_dict = self.update_metrics(
            outputs = {
                'det_boxes': det_boxes,
                'det_labels': det_labels,
                'det_scores': det_scores},
            targets={
                'gt_boxes': boxes,
                'gt_labels': labels})
        
        return loss , metric_dict

    def forward_test(self):
        inputs = torch.rand(1,3,224,224)
        if self.device:
            inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self(inputs)
        return outputs

    

    