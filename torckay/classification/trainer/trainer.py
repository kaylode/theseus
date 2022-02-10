import os
import logging
import cv2
import torch
import torchvision
from torchvision.transforms import functional as TFF
import matplotlib.pyplot as plt
from torckay.base.trainer.supervised_trainer import SupervisedTrainer
from torckay.utilities.loading import load_state_dict, find_old_tflog
from torckay.classification.augmentations.custom import Denormalize
from torckay.classification.utilities.gradcam import GradCam, show_cam_on_image
from torckay.utilities.visualization.visualizer import Visualizer
from torckay.utilities.analysis.analyzer import ClassificationAnalyzer

LOGGER = logging.getLogger("main")

class ClassificationTrainer(SupervisedTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def check_best(self, metric_dict):
        if metric_dict['bl_acc'] > self.best_value:
            LOGGER.info(f"Evaluation improved from {self.best_value} to {metric_dict['bl_acc']}")
            self.best_value = metric_dict['bl_acc']
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
        LOGGER.info("Loading checkpoints...")
        state_dict = torch.load(path)
        self.epoch = load_state_dict(self.epoch, state_dict, 'epoch')
        self.start_iter = load_state_dict(self.start_iter, state_dict, 'iters')
        self.best_value = load_state_dict(self.best_value, state_dict, 'best_value')
        
    def visualize_gt(self):
        LOGGER.debug("Visualizing dataset...")
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
        self.tf_logger.write_image(
            f'sanitycheck/batch/train', fig, step=self.iters)

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
        self.tf_logger.write_image(
            f'sanitycheck/batch/val', fig, step=self.iters)

    def visualize_pred(self):
        # Vizualize Grad Class Activation Mapping and model predictions
        LOGGER.debug("Visualizing model predictions...")

        visualizer = Visualizer()

        denom = Denormalize()
        batch = next(iter(self.valloader))
        images = batch["inputs"]
        targets = batch["targets"]

        self.model.eval()

        model_name = self.model.model.name
        grad_cam = GradCam(model=self.model, config_name=model_name)

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
        self.tf_logger.write_image(
            f'evaluation/gradcam', fig, step=self.iters)

        pred_grid_img = torchvision.utils.make_grid(pred_batch, nrow=int((idx+1)/8), normalize=False)
        fig = plt.figure(figsize=(10,10))
        plt.tight_layout(pad=0)
        plt.imshow(pred_grid_img.permute(1, 2, 0))
        plt.axis("off")
        self.tf_logger.write_image(
            f'evaluation/prediction', fig, step=self.iters)

        # Zeroing gradients in optimizer for safety
        self.optimizer.zero_grad()

    @torch.no_grad()
    def visualize_model(self):
        # Vizualize Model Graph
        LOGGER.debug("Visualizing architecture...")

        batch = next(iter(self.valloader))
        images = batch["inputs"].to(self.model.device)
        self.tf_logger.write_model(self.model.model, images)

    def analyze_gt(self):
        LOGGER.debug("Analyzing datasets...")
        analyzer = ClassificationAnalyzer()
        analyzer.add_dataset(self.trainloader.dataset)
        fig = analyzer.analyze(figsize=(10,5))
        self.tf_logger.write_image(
            f'sanitycheck/analysis/train', fig, step=0)

        analyzer = ClassificationAnalyzer()
        analyzer.add_dataset(self.valloader.dataset)
        fig = analyzer.analyze(figsize=(10,5))
        self.tf_logger.write_image(
            f'sanitycheck/analysis/val', fig, step=0)

    def on_evaluate_end(self):
        if self.visualize_when_val:
            self.visualize_pred()
        self.save_checkpoint()
    
    def on_start(self):
        if self.resume is not None:
            self.load_checkpoint(self.resume)
            self.tf_logger.load(find_old_tflog(
                os.path.dirname(os.path.dirname(self.resume))
            ))

    def sanitycheck(self):
        self.visualize_gt()
        self.analyze_gt()
        self.visualize_model()
        self.evaluate_epoch()
