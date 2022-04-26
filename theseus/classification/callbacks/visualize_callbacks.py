from typing import Dict
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import functional as TFF

from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.utilities.loggers.observer import LoggerObserver
from theseus.classification.utilities.gradcam import CAMWrapper, show_cam_on_image
from theseus.utilities.visualization.visualizer import Visualizer
from theseus.utilities.analysis.analyzer import ClassificationAnalyzer
from theseus.utilities.cuda import move_to

LOGGER = LoggerObserver.getLogger("main")

class VisualizerCallbacks(Callbacks):
    """
    Callbacks for visualizing stuff during training
    Features:
        - Visualize datasets; plot model architecture, analyze datasets in sanity check
        - Visualize prediction at every end of validation

    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.visualizer = Visualizer()

    def sanitycheck(self, logs: Dict=None):
        """
        Sanitycheck before starting. Run only when debug=True
        """

        iters = logs['iters']
        model = self.params['trainer'].model
        valloader = self.params['trainer'].valloader
        trainloader = self.params['trainer'].trainloader
        train_batch = next(iter(trainloader))
        val_batch = next(iter(valloader))
        trainset = trainloader.dataset
        valset = valloader.dataset

        self.visualize_model(model, train_batch)
        self.params['trainer'].evaluate_epoch()
        self.visualize_gt(train_batch, val_batch, iters)
        self.analyze_gt(trainset, valset, iters)

    @torch.no_grad()
    def visualize_model(self, model, batch):
        # Vizualize Model Graph
        LOGGER.text("Visualizing architecture...", level=LoggerObserver.DEBUG)
        LOGGER.log([{
            'tag': "Sanitycheck/analysis/architecture",
            'value': model.model.get_model(),
            'type': LoggerObserver.TORCH_MODULE,
            'kwargs': {
                'inputs': move_to(batch['inputs'], model.device),
                'log_freq': 100
            }
        }])

    def visualize_gt(self, train_batch, val_batch, iters):
        """
        Visualize dataloader for sanity check 
        """
        LOGGER.text("Visualizing dataset...", level=LoggerObserver.DEBUG)

        # Train batch
        images = train_batch["inputs"]

        batch = []
        for idx, inputs in enumerate(images):
            img_show = self.visualizer.denormalize(inputs)
            img_cam = TFF.to_tensor(img_show)
            batch.append(img_cam)
        grid_img = self.visualizer.make_grid(batch)

        fig = plt.figure(figsize=(8,8))
        plt.axis('off')
        plt.imshow(grid_img)
        plt.tight_layout(pad=0)
        LOGGER.log([{
            'tag': "Sanitycheck/batch/train",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': iters
            }
        }])

        # Validation batch
        images = val_batch["inputs"]

        batch = []
        for idx, inputs in enumerate(images):
            img_show = self.visualizer.denormalize(inputs)
            img_cam = TFF.to_tensor(img_show)
            batch.append(img_cam)
        grid_img = self.visualizer.make_grid(batch)

        fig = plt.figure(figsize=(8,8))
        plt.axis('off')
        plt.imshow(grid_img)
        plt.tight_layout(pad=0)

        LOGGER.log([{
            'tag': "Sanitycheck/batch/val",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': iters
            }
        }])

        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close()

    def analyze_gt(self, trainset, valset, iters):
        """
        Perform simple data analysis
        """

        LOGGER.text("Analyzing datasets...", level=LoggerObserver.DEBUG)
        analyzer = ClassificationAnalyzer()
        analyzer.add_dataset(trainset)
        fig = analyzer.analyze(figsize=(10,5))
        LOGGER.log([{
            'tag': "Sanitycheck/analysis/train",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': iters
            }
        }])

        analyzer = ClassificationAnalyzer()
        analyzer.add_dataset(valset)
        fig = analyzer.analyze(figsize=(10,5))
        LOGGER.log([{
            'tag': "Sanitycheck/analysis/val",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': iters
            }
        }])

        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close()

    @torch.enable_grad() #enable grad for CAM
    def on_val_epoch_end(self, logs: Dict=None):
        """
        After finish validation
        """

        iters = logs['iters']
        last_batch = logs['last_batch']
        model = self.params['trainer'].model
        valloader = self.params['trainer'].valloader
        optimizer = self.params['trainer'].optimizer

        # Zeroing gradients in model and optimizer for supress warning
        optimizer.zero_grad()
        model.zero_grad()

        # Vizualize Grad Class Activation Mapping and model predictions
        LOGGER.text("Visualizing model predictions...", level=LoggerObserver.DEBUG)

        images = last_batch["inputs"]
        targets = last_batch["targets"]
        model.eval()
        
        ## Calculate GradCAM
        model_name = model.model.name
        grad_cam = CAMWrapper.get_method(
            name='gradcam', 
            model=model.model.get_model(), 
            model_name=model_name, use_cuda=next(model.parameters()).is_cuda)

        grayscale_cams, label_indices, scores = grad_cam(images, return_probs=True)
            
        gradcam_batch = []
        pred_batch = []
        for idx in range(len(grayscale_cams)):
            image = images[idx]
            target = targets[idx].item()
            label = label_indices[idx]
            grayscale_cam = grayscale_cams[idx, :]
            score = scores[idx]

            img_show = self.visualizer.denormalize(image)
            self.visualizer.set_image(img_show)
            if valloader.dataset.classnames is not None:
                label = valloader.dataset.classnames[label]
                target = valloader.dataset.classnames[target]

            if label == target:
                color = [0,1,0]
            else:
                color = [1,0,0]

            self.visualizer.draw_label(
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

            pred_img = self.visualizer.get_image()
            pred_img = TFF.to_tensor(pred_img)
            pred_batch.append(pred_img)

            if idx == 63: # limit number of images
                break
        
        # GradCAM images
        gradcam_grid_img = self.visualizer.make_grid(gradcam_batch)
        fig = plt.figure(figsize=(8,8))
        plt.imshow(gradcam_grid_img)
        plt.axis("off")
        plt.tight_layout(pad=0)
        LOGGER.log([{
            'tag': "Validation/gradcam",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': iters
            }
        }])

        # Prediction images
        pred_grid_img = self.visualizer.make_grid(pred_batch)
        fig = plt.figure(figsize=(10,10))
        plt.imshow(pred_grid_img)
        plt.axis("off")
        plt.tight_layout(pad=0)
        LOGGER.log([{
            'tag': "Validation/prediction",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': iters
            }
        }])

        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close()

        # Zeroing gradients in model and optimizer for safety
        optimizer.zero_grad()
        model.zero_grad()