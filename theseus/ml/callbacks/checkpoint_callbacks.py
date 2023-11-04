import os
import os.path as osp
from typing import Dict

from theseus.base.utilities.loggers.observer import LoggerObserver
from theseus.ml.callbacks import Callbacks

LOGGER = LoggerObserver.getLogger("main")


class SKLearnCheckpointCallbacks(Callbacks):
    """
    Callbacks for saving checkpoints.
    """

    def __init__(
        self,
        save_dir: str = "runs",
        **kwargs,
    ) -> None:
        super().__init__()

        self.best_value = 0
        self.save_dir = osp.join(save_dir, "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

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
