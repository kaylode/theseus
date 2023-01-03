from typing import Dict

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import functional as TFF

from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.base.utilities.cuda import move_to
from theseus.base.utilities.loggers.observer import LoggerObserver
from theseus.tabular.base.utilities.analysis.analyzer import DataFrameAnalyzer

LOGGER = LoggerObserver.getLogger("main")


class TabularVisualizerCallbacks(Callbacks):
    """
    Callbacks for visualizing stuff during training
    Features:
        - Visualize datasets; plot model architecture, analyze datasets in sanity check
        - Visualize prediction at every end of validation

    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def sanitycheck(self, logs: Dict = None):
        """
        Sanitycheck before starting. Run only when debug=True
        """

        iters = logs["iters"]
        model = self.params["trainer"].model
        valloader = self.params["trainer"].valloader
        trainloader = self.params["trainer"].trainloader
        train_batch = next(iter(trainloader))
        trainset = trainloader.dataset
        valset = valloader.dataset

        try:
            self.visualize_model(model, train_batch)
        except TypeError as e:
            LOGGER.text("Cannot log model architecture", level=LoggerObserver.ERROR)
        self.params["trainer"].evaluate_epoch()
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

    def analyze_gt(self, trainset, valset, iters):
        """
        Perform simple data analysis
        """

        LOGGER.text("Analyzing datasets...", level=LoggerObserver.DEBUG)
        analyzer = DataFrameAnalyzer()
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

        analyzer = DataFrameAnalyzer()
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
