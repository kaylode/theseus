import os.path as osp
from typing import Dict

import torch

from theseus.base.callbacks import Callbacks
from theseus.base.utilities.loading import load_state_dict
from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


class SKLearnCheckpointCallbacks(Callbacks):
    """
    Callbacks for saving checkpoints.
    """

    def __init__(
        self,
        save_dir: str = "runs",
        save_interval: int = 10,
        **kwargs,
    ) -> None:
        super().__init__()

        self.best_value = 0
        self.save_dir = save_dir
        self.save_interval = save_interval

    def save_checkpoint(self, trainer, outname="last"):
        """
        Save all information of the current iteration
        """
        save_path = osp.join(self.save_dir, outname)
        trainer.model.save_model(savepath=save_path)
        LOGGER.text(
            f"Save model to last.pth",
            LoggerObserver.INFO,
        )

    def on_train_epoch_end(self, logs: Dict = None):
        """
        On training batch (iteration) end
        """

        # Saving checkpoint
        self.save_checkpoint(self.params["trainer"])
