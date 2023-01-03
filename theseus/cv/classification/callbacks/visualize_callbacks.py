from typing import Dict

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import functional as TFF

from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.base.utilities.cuda import move_to
from theseus.base.utilities.loggers.observer import LoggerObserver
from theseus.cv.base.utilities.analysis.analyzer import ClassificationAnalyzer
from theseus.cv.base.utilities.visualization.visualizer import Visualizer
from theseus.cv.classification.utilities.gradcam import CAMWrapper, show_cam_on_image

LOGGER = LoggerObserver.getLogger("main")


class ClassificationVisualizerCallbacks(Callbacks):
    """
    Callbacks for visualizing stuff during training
    Features:
        - Visualize datasets; plot model architecture, analyze datasets in sanity check
        - Visualize prediction at every end of validation

    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.visualizer = Visualizer()

    def sanitycheck(self, logs: Dict = None):
        """
        Sanitycheck before starting. Run only when debug=True
        """

        iters = logs["iters"]
        model = self.params["trainer"].model
        valloader = self.params["trainer"].valloader
        trainloader = self.params["trainer"].trainloader
        train_batch = next(iter(trainloader))
        val_batch = next(iter(valloader))
        trainset = trainloader.dataset
        valset = valloader.dataset

        try:
            self.visualize_model(model, train_batch)
        except TypeError as e:
            LOGGER.text("Cannot log model architecture", level=LoggerObserver.ERROR)
        self.params["trainer"].evaluate_epoch()
        self.visualize_gt(train_batch, val_batch, iters)
        self.analyze_gt(trainset, valset, iters)

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
                    "kwargs": {
                        "inputs": move_to(batch["inputs"], model.device),
                        "log_freq": 100,
                    },
                }
            ]
        )

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

        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.imshow(grid_img)
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

        # Validation batch
        images = val_batch["inputs"]

        batch = []
        for idx, inputs in enumerate(images):
            img_show = self.visualizer.denormalize(inputs)
            img_cam = TFF.to_tensor(img_show)
            batch.append(img_cam)
        grid_img = self.visualizer.make_grid(batch)

        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.imshow(grid_img)
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

    def analyze_gt(self, trainset, valset, iters):
        """
        Perform simple data analysis
        """

        LOGGER.text("Analyzing datasets...", level=LoggerObserver.DEBUG)
        analyzer = ClassificationAnalyzer()
        analyzer.add_dataset(trainset)
        fig = analyzer.analyze(figsize=(10, 5))
        LOGGER.log(
            [
                {
                    "tag": "Sanitycheck/analysis/train",
                    "value": fig,
                    "type": LoggerObserver.FIGURE,
                    "kwargs": {"step": iters},
                }
            ]
        )

        analyzer = ClassificationAnalyzer()
        analyzer.add_dataset(valset)
        fig = analyzer.analyze(figsize=(10, 5))
        LOGGER.log(
            [
                {
                    "tag": "Sanitycheck/analysis/val",
                    "value": fig,
                    "type": LoggerObserver.FIGURE,
                    "kwargs": {"step": iters},
                }
            ]
        )

        plt.cla()  # Clear axis
        plt.clf()  # Clear figure
        plt.close()

    @torch.no_grad()  # enable grad for CAM
    def on_val_epoch_end(self, logs: Dict = None):
        """
        After finish validation
        """

        iters = logs["iters"]
        last_batch = logs["last_batch"]
        model = self.params["trainer"].model
        valloader = self.params["trainer"].valloader

        # Vizualize model predictions
        LOGGER.text("Visualizing model predictions...", level=LoggerObserver.DEBUG)

        images = last_batch["inputs"]
        targets = last_batch["targets"]
        model.eval()

        ## Get prediction on last batch
        outputs = model.model.get_prediction(last_batch, device=model.device)
        label_indices = outputs["labels"]
        scores = outputs["confidences"]

        pred_batch = []
        for idx in range(len(images)):
            image = images[idx]
            target = targets[idx].item()
            label = label_indices[idx]
            score = scores[idx]

            img_show = self.visualizer.denormalize(image)
            self.visualizer.set_image(img_show)
            if valloader.dataset.classnames is not None:
                label = valloader.dataset.classnames[label]
                target = valloader.dataset.classnames[target]

            if label == target:
                color = [0, 1, 0]
            else:
                color = [1, 0, 0]

            self.visualizer.draw_label(
                f"GT: {target}\nP: {label}\nC: {score:.4f}",
                fontColor=color,
                fontScale=0.8,
                thickness=2,
                outline=None,
                offset=100,
            )

            pred_img = self.visualizer.get_image()
            pred_img = TFF.to_tensor(pred_img)
            pred_batch.append(pred_img)

            if idx == 63:  # limit number of images
                break

        # Prediction images
        pred_grid_img = self.visualizer.make_grid(pred_batch)
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(pred_grid_img)
        plt.axis("off")
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
