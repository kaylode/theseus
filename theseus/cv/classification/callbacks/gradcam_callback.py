from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.transforms import functional as TFF

from theseus.base.utilities.loggers.observer import LoggerObserver
from theseus.cv.base.utilities.visualization.visualizer import Visualizer
from theseus.cv.classification.utilities.gradcam import CAMWrapper, show_cam_on_image

LOGGER = LoggerObserver.getLogger("main")


class GradCAMVisualizationCallback(Callback):
    """
    Callbacks for visualizing stuff during training
    Features:
        - Visualize datasets; plot model architecture, analyze datasets in sanity check
        - Visualize prediction at every end of validation

    """

    def __init__(
        self,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        **kwargs
    ) -> None:
        super().__init__()
        self.visualizer = Visualizer()
        self.mean = mean
        self.std = std

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.params = {}
        self.params["last_batch"] = batch

    @torch.enable_grad()  # enable grad for CAM
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        After finish validation
        """

        iters = trainer.global_step
        last_batch = self.params["last_batch"]
        model = pl_module.model
        valloader = pl_module.datamodule.valloader
        optimizer = pl_module.optimizer

        # Zeroing gradients in model and optimizer for supress warning
        optimizer.zero_grad()
        model.zero_grad()

        # Vizualize Grad Class Activation Mapping and model predictions
        LOGGER.text("Visualizing model predictions...", level=LoggerObserver.DEBUG)

        images = last_batch["inputs"].cpu()
        targets = last_batch["targets"]
        model.eval()

        ## Calculate GradCAM and Grad Class Activation Mapping and
        model_name = model.name

        try:
            grad_cam = CAMWrapper.get_method(
                name="gradcam",
                model=model.get_model(),
                model_name=model_name,
                use_cuda=next(model.parameters()).is_cuda,
            )

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

            img_show = self.visualizer.denormalize(image, mean=self.mean, std=self.std)
            if valloader.dataset.classnames is not None:
                label = valloader.dataset.classnames[label]
                target = valloader.dataset.classnames[target]

            img_cam = show_cam_on_image(img_show, grayscale_cam, use_rgb=True)

            img_cam = TFF.to_tensor(img_cam)
            gradcam_batch.append(img_cam)

            if idx == 63:  # limit number of images
                break

        # GradCAM images
        gradcam_grid_img = self.visualizer.make_grid(gradcam_batch)
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(gradcam_grid_img)
        plt.axis("off")
        plt.tight_layout(pad=0)
        LOGGER.log(
            [
                {
                    "tag": "Validation/gradcam",
                    "value": fig,
                    "type": LoggerObserver.FIGURE,
                    "kwargs": {"step": iters},
                }
            ]
        )
