from typing import Any, Dict

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


class SemanticVisualizerCallback(Callback):
    """
    Callbacks for visualizing stuff during training
    Features:
        - Visualize datasets; plot model architecture, analyze datasets in sanity check
        - Visualize prediction at every end of validation

    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.visualizer = Visualizer()

    def on_sanity_check_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Sanitycheck before starting. Run only when debug=True
        """

        iters = trainer.global_step
        model = pl_module.model
        valloader = pl_module.datamodule.valloader
        trainloader = pl_module.datamodule.trainloader
        train_batch = next(iter(trainloader))
        val_batch = next(iter(valloader))
        valset = valloader.dataset
        classnames = valset.classnames

        try:
            self.visualize_model(model, train_batch)
        except:
            LOGGER.text("Cannot log model architecture", level=LoggerObserver.ERROR)
        self.visualize_gt(train_batch, val_batch, iters, classnames)

    @torch.no_grad()
    def visualize_model(self, model, batch):
        # Vizualize Model Graph
        LOGGER.text("Visualizing architecture...", level=LoggerObserver.DEBUG)
        LOGGER.log(
            [
                {
                    "tag": "Sanitycheck/analysis/architecture",
                    "value": model.model.get_model(),
                    "type": LoggerObserver.TORCH_MODULE,
                    "kwargs": {"inputs": batch, "log_freq": 100},
                }
            ]
        )

    def visualize_gt(self, train_batch, val_batch, iters, classnames):
        """
        Visualize dataloader for sanity check
        """

        LOGGER.text("Visualizing dataset...", level=LoggerObserver.DEBUG)
        images = train_batch["inputs"]
        masks = train_batch["targets"].squeeze()

        batch = []
        for idx, (inputs, mask) in enumerate(zip(images, masks)):
            img_show = self.visualizer.denormalize(inputs)
            decode_mask = self.visualizer.decode_segmap(mask.numpy())
            img_show = TFF.to_tensor(img_show)
            decode_mask = TFF.to_tensor(decode_mask / 255.0)
            img_show = torch.cat([img_show, decode_mask], dim=-1)
            batch.append(img_show)
        grid_img = self.visualizer.make_grid(batch)

        fig = plt.figure(figsize=(16, 8))
        plt.axis("off")
        plt.imshow(grid_img)

        # segmentation color legends
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
                    "tag": "Sanitycheck/batch/train",
                    "value": fig,
                    "type": LoggerObserver.FIGURE,
                    "kwargs": {"step": iters},
                }
            ]
        )

        # Validation
        images = val_batch["inputs"]
        masks = val_batch["targets"].squeeze()

        batch = []
        for idx, (inputs, mask) in enumerate(zip(images, masks)):
            img_show = self.visualizer.denormalize(inputs)
            decode_mask = self.visualizer.decode_segmap(mask.numpy())
            img_show = TFF.to_tensor(img_show)
            decode_mask = TFF.to_tensor(decode_mask / 255.0)
            img_show = torch.cat([img_show, decode_mask], dim=-1)
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
        masks = last_batch["targets"].squeeze()

        preds = model.model.get_prediction({"inputs": images}, model.device)["masks"]

        batch = []
        for idx, (inputs, mask, pred) in enumerate(zip(images, masks, preds)):
            img_show = self.visualizer.denormalize(inputs)
            decode_mask = self.visualizer.decode_segmap(mask.numpy())
            decode_pred = self.visualizer.decode_segmap(pred)
            img_cam = TFF.to_tensor(img_show)
            decode_mask = TFF.to_tensor(decode_mask / 255.0)
            decode_pred = TFF.to_tensor(decode_pred / 255.0)
            img_show = torch.cat([img_cam, decode_pred, decode_mask], dim=-1)
            batch.append(img_show)
        grid_img = self.visualizer.make_grid(batch)

        fig = plt.figure(figsize=(16, 8))
        plt.axis("off")
        plt.title("Raw image - Prediction - Ground Truth")
        plt.imshow(grid_img)

        # segmentation color legends
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
