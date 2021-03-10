from .base_model import BaseModel
import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm


import sys
sys.path.append('..')

class Detector(BaseModel):
    def __init__(self, model, n_classes, **kwargs):
        super(Detector, self).__init__(**kwargs)
        self.model = model
        self.model_name = self.model.name
        if self.optimizer is not None:
            self.optimizer = self.optimizer(self.parameters(), lr= self.lr)
            self.set_optimizer_params()
        self.n_classes = n_classes

        if self.freeze:
            for params in self.model.parameters():
                params.requires_grad = False

        if self.device:
            self.model.to(self.device)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        inputs = batch["imgs"]
        targets = batch['targets']
        targets_res = {}
        if self.device:
            inputs = inputs.to(self.device)
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]

        targets_res['boxes'] = boxes
        targets_res['labels'] = labels
        output = self.model(inputs, targets_res)

        loss_dict = {k:v.item() for k,v in output.items()}
        loss = output['T']
        return loss, loss_dict

    
    def inference_step(self, batch, conf_threshold = 0.001, iou_threshold=0.5):
        inputs = batch["imgs"]
        img_sizes = batch['img_sizes']
        img_scales = batch['img_scales']

        if self.device:
            inputs = inputs.to(self.device)
            img_sizes = img_sizes.to(self.device)
            img_scales = img_scales.to(self.device)
        outputs = self.model.detect(inputs, img_sizes, img_scales, conf_threshold=conf_threshold)
            
        return outputs  

    def evaluate_step(self, batch):
        inputs = batch["imgs"]
        targets = batch['targets']
        targets_res = {}
        if self.device:
            inputs = inputs.to(self.device)
            boxes = [target['boxes'].to(self.device) for target in targets]
            labels = [target['labels'].to(self.device) for target in targets]

        targets_res['boxes'] = boxes
        targets_res['labels'] = labels
        output = self.model(inputs, targets_res)

        loss_dict = {k:v.item() for k,v in output.items()}
        loss = output['T']
        
        self.update_metrics(model=self)
        
        return loss, loss_dict

    def forward_test(self, size = 224):
        inputs = torch.rand(1,3,size,size)
        if self.device:
            inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self(inputs)
        return outputs

    