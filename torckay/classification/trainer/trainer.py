import os
import cv2
import torch
import torchvision
from torchvision.transforms import functional as TFF
import matplotlib.pyplot as plt
from torckay.base.trainer.supervised_trainer import SupervisedTrainer
from torckay.utilities.loggers.logger import LoggerManager
from torckay.utilities.loading import load_state_dict
from torckay.classification.augmentations.custom import Denormalize
from torckay.classification.utilities.gradcam import GradCam, show_cam_on_image

LOGGER = LoggerManager.init_logger(__name__)

class ClassificationTrainer(SupervisedTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def check_best(self, metric_dict):
        if metric_dict['acc'] > self.best_value:
            self.save_checkpoint('best')

    def save_checkpoint(self, outname='last'):
        weights = {
            'model': self.model.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'iters': self.iters,
            'best_value': self.best_value,
        }

        if self.scaler is not None:
            weights[self.scaler.state_dict_key] = self.scaler.state_dict()
           
        self.checkpoint.save(weights, outname)

    def load_checkpoint(self, path):
        state_dict = torch.load(path)
        self.model.model = load_state_dict(self.model.model, state_dict, 'model')
        self.optimizer = load_state_dict(self.optimizer, state_dict, 'optimizer')
        self.scaler = load_state_dict(self.scaler, state_dict, self.scaler.state_dict_key)
        self.epoch = load_state_dict(self.epoch, state_dict, 'epoch')
        self.start_iter = load_state_dict(self.start_iter, state_dict, 'iters')
        self.best_value = load_state_dict(self.best_value, state_dict, 'best_value')
        
    def visualize_batch(self):
        # Vizualize Grad Class Activation Mapping
        denom = Denormalize()
        batch = next(iter(self.valloader))
        images = batch["inputs"]

        self.model.eval()

        model_name = self.model.model.name
        grad_cam = GradCam(model=self.model, config_name=model_name)

        batch = []
        for idx, inputs in enumerate(images):
            img_show = denom(inputs)
            inputs = inputs.unsqueeze(0)
            inputs = inputs.to(self.model.device)
            target_category = None
            grayscale_cam, label_idx = grad_cam(inputs, target_category)
            label = self.val_dataset.classnames[label_idx]
            img_cam = show_cam_on_image(img_show, grayscale_cam, label)
            img_cam = TFF.to_tensor(img_cam)
            batch.append(img_cam)

        batch = torch.stack(batch, dim=0)
        grid_img = torchvision.utils.make_grid(batch, nrow=8, normalize=False)

        self.tf_logger.write_image(
            f'samples/{self.epoch}_{self.iters}', grid_img.permute(1, 2, 0), step=self.epoch)


    def on_evaluate_end(self):
        if self.visualize_when_val:
            self.visualize_batch()
        self.save_checkpoint()
    
    def on_training_start(self):
        if self.resume is not None:
            self.load_checkpoint(self.resume)
        

    def on_training_end(self):
        return

    def on_evaluate_end(self):
        return