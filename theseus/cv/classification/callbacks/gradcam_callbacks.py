from typing import Dict
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import functional as TFF

from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.base.utilities.loggers.observer import LoggerObserver
from theseus.cv.classification.utilities.gradcam import CAMWrapper, show_cam_on_image
from theseus.cv.base.utilities.visualization.visualizer import Visualizer

LOGGER = LoggerObserver.getLogger("main")

class GradCAMVisualizationCallbacks(Callbacks):
    """
    Callbacks for visualizing stuff during training
    Features:
        - Visualize datasets; plot model architecture, analyze datasets in sanity check
        - Visualize prediction at every end of validation

    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.visualizer = Visualizer()

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
        
        ## Calculate GradCAM and Grad Class Activation Mapping and
        model_name = model.model.name

        try:
            grad_cam = CAMWrapper.get_method(
                name='gradcam', 
                model=model.model.get_model(), 
                model_name=model_name, use_cuda=next(model.parameters()).is_cuda)
            
            grayscale_cams, label_indices, scores = grad_cam(images, return_probs=True)

        except:
            LOGGER.text("Cannot calculate GradCAM", level=LoggerObserver.ERROR)
            return
            
        gradcam_batch = []
        for idx in range(len(grayscale_cams)):
            image = images[idx]
            target = targets[idx].item()
            label = label_indices[idx]
            grayscale_cam = grayscale_cams[idx, :]

            img_show = self.visualizer.denormalize(image)
            if valloader.dataset.classnames is not None:
                label = valloader.dataset.classnames[label]
                target = valloader.dataset.classnames[target]

            img_cam =show_cam_on_image(img_show, grayscale_cam, use_rgb=True)

            img_cam = TFF.to_tensor(img_cam)
            gradcam_batch.append(img_cam)

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