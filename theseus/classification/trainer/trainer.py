import torch
from torchvision.transforms import functional as TFF
import matplotlib.pyplot as plt
from theseus.base.trainer.supervised_trainer import SupervisedTrainer
from theseus.utilities.loading import load_state_dict
from theseus.classification.utilities.gradcam import CAMWrapper, show_cam_on_image
from theseus.utilities.visualization.visualizer import Visualizer
from theseus.utilities.analysis.analyzer import ClassificationAnalyzer

from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

class ClassificationTrainer(SupervisedTrainer):
    """Trainer for classification tasks
    
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def check_best(self, metric_dict):
        """
        Hook function, called after metrics are calculated
        """
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
        """
        Save all information of the current iteration
        """
        weights = {
            'model': self.model.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iters': self.iters,
            'best_value': self.best_value,
        }

        if self.scaler is not None:
            weights[self.scaler.state_dict_key] = self.scaler.state_dict()
           
        self.checkpoint.save(weights, outname)

    def load_checkpoint(self, path:str):
        """
        Load all information the current iteration from checkpoint 
        """
        LOGGER.text("Loading checkpoints...", level=LoggerObserver.INFO)
        state_dict = torch.load(path, map_location='cpu')
        self.iters = load_state_dict(self.iters, state_dict, 'iters')
        self.best_value = load_state_dict(self.best_value, state_dict, 'best_value')  
        self.scaler = load_state_dict(self.scaler, state_dict, self.scaler.state_dict_key)

        
    def visualize_gt(self):
        """
        Visualize dataloader for sanity check 
        """

        LOGGER.text("Visualizing dataset...", level=LoggerObserver.DEBUG)
        visualizer = Visualizer()

        # Train batch
        batch = next(iter(self.trainloader))
        images = batch["inputs"]

        batch = []
        for idx, inputs in enumerate(images):
            img_show = visualizer.denormalize(inputs)
            img_cam = TFF.to_tensor(img_show)
            batch.append(img_cam)
        grid_img = visualizer.make_grid(batch)

        fig = plt.figure(figsize=(8,8))
        plt.axis('off')
        plt.imshow(grid_img)
        plt.tight_layout(pad=0)
        LOGGER.log([{
            'tag': "Sanitycheck/batch/train",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': self.iters
            }
        }])

        # Validation batch
        batch = next(iter(self.valloader))
        images = batch["inputs"]

        batch = []
        for idx, inputs in enumerate(images):
            img_show = visualizer.denormalize(inputs)
            img_cam = TFF.to_tensor(img_show)
            batch.append(img_cam)
        grid_img = visualizer.make_grid(batch)

        fig = plt.figure(figsize=(8,8))
        plt.axis('off')
        plt.imshow(grid_img)
        plt.tight_layout(pad=0)

        LOGGER.log([{
            'tag': "Sanitycheck/batch/val",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': self.iters
            }
        }])


    @torch.enable_grad() #enable grad for CAM
    def visualize_pred(self):
        r"""Visualize model prediction and CAM
        
        """
        # Vizualize Grad Class Activation Mapping and model predictions
        LOGGER.text("Visualizing model predictions...", level=LoggerObserver.DEBUG)

        visualizer = Visualizer()

        batch = next(iter(self.valloader))
        images = batch["inputs"]
        targets = batch["targets"]

        self.model.eval()

        model_name = self.model.model.name
        grad_cam = CAMWrapper.get_method(
            name='gradcam', 
            model=self.model.model.get_model(), 
            model_name=model_name, use_cuda=next(self.model.parameters()).is_cuda)

        grayscale_cams, label_indices, scores = grad_cam(images, return_probs=True)
            
        gradcam_batch = []
        pred_batch = []
        for idx in range(len(grayscale_cams)):
            image = images[idx]
            target = targets[idx].item()
            label = label_indices[idx]
            grayscale_cam = grayscale_cams[idx, :]
            score = scores[idx]

            img_show = visualizer.denormalize(image)
            visualizer.set_image(img_show)
            if self.valloader.dataset.classnames is not None:
                label = self.valloader.dataset.classnames[label]
                target = self.valloader.dataset.classnames[target]

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
            
            img_cam =show_cam_on_image(img_show, grayscale_cam, use_rgb=True)

            img_cam = TFF.to_tensor(img_cam)
            gradcam_batch.append(img_cam)

            pred_img = visualizer.get_image()
            pred_img = TFF.to_tensor(pred_img)
            pred_batch.append(pred_img)

            if idx == 63: # limit number of images
                break
        
        # GradCAM images
        gradcam_grid_img = visualizer.make_grid(gradcam_batch)
        fig = plt.figure(figsize=(8,8))
        plt.imshow(gradcam_grid_img)
        plt.axis("off")
        plt.tight_layout(pad=0)
        LOGGER.log([{
            'tag': "Validation/gradcam",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': self.iters
            }
        }])

        # Prediction images
        pred_grid_img = visualizer.make_grid(pred_batch)
        fig = plt.figure(figsize=(10,10))
        plt.imshow(pred_grid_img)
        plt.axis("off")
        plt.tight_layout(pad=0)
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
            'value': self.model.model.get_model(),
            'type': LoggerObserver.TORCH_MODULE,
            'kwargs': {
                'inputs': images
            }
        }])

    def analyze_gt(self):
        """
        Perform simple data analysis
        """
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
        """Sanity check before training
        """
        self.visualize_gt()
        self.analyze_gt()
        self.visualize_model()
        self.evaluate_epoch()
