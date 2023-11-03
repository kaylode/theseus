from typing import Any, Dict, List

import lightning.pytorch as pl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.transforms import functional as TFF

from theseus.base.utilities.loggers.observer import LoggerObserver
from theseus.cv.base.utilities.visualization.colors import color_list
from theseus.cv.base.utilities.visualization.visualizer import Visualizer

LOGGER = LoggerObserver.getLogger("main")


class DetectionVisualizerCallback(Callback):
    """
    Callbacks for visualizing stuff during training
    Features:
        - Visualize datasets; plot model architecture, analyze datasets in sanity check
        - Visualize prediction at every end of validation
    """

    def __init__(
        self,
        order: str = None,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        **kwargs
    ) -> None:
        super().__init__()
        self.visualizer = Visualizer()
        self.order = order
        self.mean = mean
        self.std = std

    def on_sanity_check_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Sanitycheck before starting. Run only when debug=True
        """

        iters = trainer.iterations
        model = pl_module.model
        valloader = pl_module.datamodule.valloader
        trainloader = pl_module.datamodule.trainloader
        train_batch = next(iter(trainloader))
        val_batch = next(iter(valloader))
        valset = valloader.dataset
        classnames = valset.classnames

        self.visualizer.set_classnames(classnames)
        self.visualize_gt(train_batch, val_batch, iters, classnames)

    def visualize_gt(self, train_batch, val_batch, iters, classnames):
        """
        Visualize dataloader for sanity check
        """

        LOGGER.text("Visualizing dataset...", level=LoggerObserver.DEBUG)
        images = train_batch["inputs"]
        anns = train_batch["targets"]

        batch = []
        for idx, (inputs, ann) in enumerate(zip(images, anns)):
            boxes = ann["boxes"]
            labels = ann["labels"].numpy()
            img_show = self.visualizer.denormalize(inputs, mean=self.mean, std=self.std)
            decode_boxes = self.visualizer.denormalize_bboxes(
                boxes, order=self.order, image_shape=img_show.shape[:2], auto_scale=True
            )

            self.visualizer.set_image(img_show.copy())
            self.visualizer.draw_bbox(decode_boxes, labels=labels)
            img_show = self.visualizer.get_image()
            img_show = TFF.to_tensor(img_show)
            batch.append(img_show)
        grid_img = self.visualizer.make_grid(batch)

        fig = plt.figure(figsize=(16, 8))
        plt.axis("off")
        plt.imshow(grid_img)

        # segmentation color legends
        patches = [
            mpatches.Patch(color=np.array(color_list[i]), label=classnames[i])
            for i in range(len(classnames))
        ]
        plt.legend(
            handles=patches,
            bbox_to_anchor=(-0.03, 1),
            loc="upper right",
            borderaxespad=0.0,
            fontsize="large",
            ncol=(len(classnames) // 10) + 1,
        )
        plt.tight_layout(pad=0)

        LOGGER.log(
            [
                {
                    "tag": "Sanitycheck/batch/train",
                    "value": fig,
                    "type": LoggerObserver.FIGURE,
                    "kwargs": {"step": iters},
                }
            ]
        )

        # Validation
        images = val_batch["inputs"]
        anns = val_batch["targets"]

        batch = []
        for idx, (inputs, ann) in enumerate(zip(images, anns)):
            boxes = ann["boxes"]
            labels = ann["labels"].numpy()
            img_show = self.visualizer.denormalize(inputs, mean=self.mean, std=self.std)
            decode_boxes = self.visualizer.denormalize_bboxes(
                boxes, order=self.order, image_shape=img_show.shape[:2], auto_scale=True
            )

            self.visualizer.set_image(img_show.copy())
            self.visualizer.draw_bbox(decode_boxes, labels=labels)
            img_show = self.visualizer.get_image()
            img_show = TFF.to_tensor(img_show)
            batch.append(img_show)
        grid_img = self.visualizer.make_grid(batch)

        fig = plt.figure(figsize=(16, 8))
        plt.axis("off")
        plt.imshow(grid_img)
        plt.legend(
            handles=patches,
            bbox_to_anchor=(-0.03, 1),
            loc="upper right",
            borderaxespad=0.0,
            fontsize="large",
            ncol=(len(classnames) // 10) + 1,
        )
        plt.tight_layout(pad=0)

        LOGGER.log(
            [
                {
                    "tag": "Sanitycheck/batch/val",
                    "value": fig,
                    "type": LoggerObserver.FIGURE,
                    "kwargs": {"step": iters},
                }
            ]
        )

        plt.cla()  # Clear axis
        plt.clf()  # Clear figure
        plt.close()

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

    @torch.no_grad()
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        After finish validation
        """

        iters = trainer.global_step
        last_batch = self.params["last_batch"]
        model = pl_module.model
        valloader = pl_module.datamodule.valloader

        # Vizualize model predictions
        LOGGER.text("Visualizing model predictions...", level=LoggerObserver.DEBUG)

        model.eval()

        images = last_batch["inputs"]
        targets = last_batch["targets"]

        preds = model.model.get_prediction({"inputs": images}, model.device)

        preds = [i for i in zip(preds["boxes"], preds["confidences"], preds["labels"])]

        batch = []
        for idx, (inputs, target, pred) in enumerate(zip(images, targets, preds)):
            # Ground truth
            boxes = target["boxes"]
            labels = target["labels"].numpy()
            img_show = self.visualizer.denormalize(inputs, mean=self.mean, std=self.std)
            self.visualizer.set_image(img_show.copy())
            self.visualizer.draw_bbox(boxes, labels=labels)
            img_show = self.visualizer.get_image()
            img_show = TFF.to_tensor(img_show / 255.0)

            # Prediction
            boxes, scores, labels = pred
            decode_pred = self.visualizer.denormalize(
                inputs, mean=self.mean, std=self.std
            )
            self.visualizer.set_image(decode_pred.copy())
            self.visualizer.draw_bbox(boxes, labels=labels, scores=scores)
            decode_pred = self.visualizer.get_image()
            decode_pred = TFF.to_tensor(decode_pred / 255.0)

            img_show = torch.cat([img_show, decode_pred], dim=-1)
            batch.append(img_show)
        grid_img = self.visualizer.make_grid(batch)

        fig = plt.figure(figsize=(16, 8))
        plt.axis("off")
        plt.title("Raw image - Ground Truth - Prediction")
        plt.imshow(grid_img)

        # color legends
        classnames = valloader.dataset.classnames
        patches = [
            mpatches.Patch(color=np.array(color_list[i][::-1]), label=classnames[i])
            for i in range(len(classnames))
        ]
        plt.legend(
            handles=patches,
            bbox_to_anchor=(-0.03, 1),
            loc="upper right",
            borderaxespad=0.0,
            fontsize="large",
            ncol=(len(classnames) // 10) + 1,
        )
        plt.tight_layout(pad=0)

        LOGGER.log(
            [
                {
                    "tag": "Validation/prediction",
                    "value": fig,
                    "type": LoggerObserver.FIGURE,
                    "kwargs": {"step": iters},
                }
            ]
        )

        plt.cla()  # Clear axis
        plt.clf()  # Clear figure
        plt.close()
