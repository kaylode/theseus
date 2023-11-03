import json
import os
import os.path as osp
from typing import Dict

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


class MetricLoggerCallback(Callback):
    """
    Callbacks for logging running metric while training every epoch end
    Features:
        - Only do logging
    """

    def __init__(self, save_json: bool = True, **kwargs) -> None:
        super().__init__()
        self.save_json = save_json
        if self.save_json:
            self.save_dir = kwargs.get("save_dir", None)
            if self.save_dir is not None:
                self.save_dir = osp.join(self.save_dir, "Validation")
                os.makedirs(self.save_dir, exist_ok=True)
            self.output_dict = []

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        After finish validation
        """
        iters = trainer.global_step
        metric_dict = pl_module.metric_dict

        # Save json
        if self.save_json:
            item = {}
            for metric, score in metric_dict.items():
                if isinstance(score, (int, float)):
                    item[metric] = float(f"{score:.5f}")
            if len(item.keys()) > 0:
                item["iters"] = iters
                self.output_dict.append(item)

        # Log metric
        metric_string = ""
        for metric, score in metric_dict.items():
            if isinstance(score, (int, float)):
                metric_string += metric + ": " + f"{score:.5f}" + " | "
        metric_string += "\n"

        LOGGER.text(metric_string, level=LoggerObserver.INFO)

        # Call other loggers
        log_dict = [
            {"tag": f"Validation/{k}", "value": v, "kwargs": {"step": iters}}
            for k, v in metric_dict.items()
        ]

        LOGGER.log(log_dict)

    def teardown(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """
        After finish everything
        """
        if self.save_json:
            save_json = osp.join(self.save_dir, "metrics.json")
            if len(self.output_dict) > 0:
                with open(save_json, "w") as f:
                    json.dump(self.output_dict, f)
