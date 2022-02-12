import os
import logging
import cv2
import torch
import torchvision
from torchvision.transforms import functional as TFF
import matplotlib.pyplot as plt
from torckay.base.trainer.supervised_trainer import SupervisedTrainer
from torckay.utilities.loading import load_state_dict
from torckay.base.augmentations.custom import Denormalize
from torckay.classification.utilities.gradcam import GradCam, show_cam_on_image
from torckay.utilities.visualization.visualizer import Visualizer
from torckay.utilities.analysis.analyzer import ClassificationAnalyzer

from torckay.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

class ClassificationTrainer(SupervisedTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def check_best(self, metric_dict):
        if metric_dict['bl_acc'] > self.best_value:
            if self.iters > 0: # Have been training, else in evaluation-only mode or just sanity check
                LOGGER.text(
                    f"Evaluation improved from {self.best_value} to {metric_dict['bl_acc']}",
                    level=LoggerObserver.INFO)
                self.best_value = metric_dict['bl_acc']
                self.save_checkpoint('best')
            
            else:
                if self.visualize_when_val:
                    self.visualize_pred()

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
        LOGGER.text("Loading checkpoints...", level=LoggerObserver.INFO)
        state_dict = torch.load(path)
        self.epoch = load_state_dict(self.epoch, state_dict, 'epoch')
        self.start_iter = load_state_dict(self.start_iter, state_dict, 'iters')
        self.best_value = load_state_dict(self.best_value, state_dict, 'best_value')  
        self.scaler = load_state_dict(self.scaler, state_dict, self.scaler.state_dict_key)

        
    def visualize_gt(self):
        LOGGER.text("Visualizing dataset...", level=LoggerObserver.DEBUG)
        denom = Denormalize()
        batch = next(iter(self.trainloader))
        images = batch["inputs"]

        batch = []
        for idx, inputs in enumerate(images):
            img_show = denom(inputs)
            img_cam = TFF.to_tensor(img_show)
            batch.append(img_cam)
        batch = torch.stack(batch, dim=0)
        grid_img = torchvision.utils.make_grid(batch, nrow=int((idx+1)/8), normalize=False)

        fig = plt.figure(figsize=(8,8))
        plt.tight_layout(pad=0)
        plt.axis('off')
        plt.imshow(grid_img.permute(1, 2, 0))
        LOGGER.log([{
            'tag': "Sanitycheck/batch/train",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': self.iters
            }
        }])

        

        batch = next(iter(self.valloader))
        images = batch["inputs"]

        batch = []
        for idx, inputs in enumerate(images):
            img_show = denom(inputs)
            img_cam = TFF.to_tensor(img_show)
            batch.append(img_cam)
        batch = torch.stack(batch, dim=0)
        grid_img = torchvision.utils.make_grid(batch, nrow=int((idx+1)/8), normalize=False)

        fig = plt.figure(figsize=(8,8))
        plt.tight_layout(pad=0)
        plt.axis('off')
        plt.imshow(grid_img.permute(1, 2, 0))

        LOGGER.log([{
            'tag': "Sanitycheck/batch/val",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': self.iters
            }
        }])


    @torch.enable_grad() #enable grad for GradCAM
    def visualize_pred(self):
        # Vizualize Grad Class Activation Mapping and model predictions
        LOGGER.text("Visualizing model predictions...", level=LoggerObserver.DEBUG)

        visualizer = Visualizer()

        denom = Denormalize()
        batch = next(iter(self.valloader))
        images = batch["inputs"]
        targets = batch["targets"]

        self.model.eval()

        model_name = self.model.model.name
        grad_cam = GradCam(model=self.model.model.get_model(), config_name=model_name)

        gradcam_batch = []
        pred_batch = []
        for idx, (input, target) in enumerate(zip(images, targets)):
            img_show = denom(input)
            visualizer.set_image(img_show)
            input = input.unsqueeze(0)
            input = input.to(self.model.device)
            target_category = None
            grayscale_cam, label_idx, score = grad_cam(input, target_category, return_prob=True)
            label = self.valloader.dataset.classnames[label_idx]
            target = self.valloader.dataset.classnames[target.item()]

            if label == target:
                color = [0,1,0]
            else:
                color = [1,0,0]

            visualizer.draw_label(
                f"GT: {target}\nP: {label}\nC: {score:.4f}", 
                fontColor=color, 
                fontScale=0.8,
                thickness=2,
                outline=None,
                offset=100
            )

            img_cam = show_cam_on_image(img_show, grayscale_cam, label)
            img_cam = cv2.cvtColor(img_cam, cv2.COLOR_BGR2RGB)
            img_cam = TFF.to_tensor(img_cam)
            gradcam_batch.append(img_cam)

            pred_img = visualizer.get_image()
            pred_img = TFF.to_tensor(pred_img)
            pred_batch.append(pred_img)

            if idx == 63: # limit number of images
                break

        gradcam_batch = torch.stack(gradcam_batch, dim=0)
        pred_batch = torch.stack(pred_batch, dim=0)

        gradcam_grid_img = torchvision.utils.make_grid(gradcam_batch, nrow=int((idx+1)/8), normalize=False)

        fig = plt.figure(figsize=(8,8))
        plt.tight_layout(pad=0)
        plt.imshow(gradcam_grid_img.permute(1, 2, 0))
        plt.axis("off")

        LOGGER.log([{
            'tag': "Validation/gradcam",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': self.iters
            }
        }])

        pred_grid_img = torchvision.utils.make_grid(pred_batch, nrow=int((idx+1)/8), normalize=False)
        fig = plt.figure(figsize=(10,10))
        plt.tight_layout(pad=0)
        plt.imshow(pred_grid_img.permute(1, 2, 0))
        plt.axis("off")

        LOGGER.log([{
            'tag': "Validation/prediction",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': self.iters
            }
        }])

        # Zeroing gradients in optimizer for safety
        self.optimizer.zero_grad()

    @torch.no_grad()
    def visualize_model(self):
        # Vizualize Model Graph
        LOGGER.text("Visualizing architecture...", level=LoggerObserver.DEBUG)

        batch = next(iter(self.valloader))
        images = batch["inputs"].to(self.model.device)
        LOGGER.log([{
            'tag': "Sanitycheck/analysis/architecture",
            'value': self.model.model,
            'type': LoggerObserver.TORCH_MODULE,
            'kwargs': {
                'inputs': images
            }
        }])

    def analyze_gt(self):
        LOGGER.text("Analyzing datasets...", level=LoggerObserver.DEBUG)
        analyzer = ClassificationAnalyzer()
        analyzer.add_dataset(self.trainloader.dataset)
        fig = analyzer.analyze(figsize=(10,5))
        LOGGER.log([{
            'tag': "Sanitycheck/analysis/train",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': self.iters
            }
        }])

        analyzer = ClassificationAnalyzer()
        analyzer.add_dataset(self.valloader.dataset)
        fig = analyzer.analyze(figsize=(10,5))
        LOGGER.log([{
            'tag': "Sanitycheck/analysis/val",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': self.iters
            }
        }])

    def on_evaluate_end(self):
        if self.visualize_when_val:
            self.visualize_pred()
        self.save_checkpoint()
    
    def on_start(self):
        if self.resume is not None:
            self.load_checkpoint(self.resume)

    def sanitycheck(self):
        self.visualize_gt()
        self.analyze_gt()
        self.visualize_model()
        self.evaluate_epoch()
