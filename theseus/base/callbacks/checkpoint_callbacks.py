from distutils.log import WARN
from typing import List, Dict
import os.path as osp

from theseus.base.callbacks import Callbacks
from theseus.utilities.loggers.cp_logger import Checkpoint
from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

class CheckpointCallbacks(Callbacks):
    """
    Callbacks for saving checkpoints
    """

    def __init__(self, save_dir: str, save_interval: int=10, best_key:str = None) -> None:
        super().__init__()

        self.best_value = 0
        self.best_key = best_key
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.checkpoint = Checkpoint(osp.join(self.save_dir, 'checkpoints')) 

    def on_train_batch_end(self, logs:Dict=None):
        """
        On training batch (iteration) end
        """

        iters = logs['iters']
        num_iterations = logs['num_iterations']

        # Saving checkpoint
        if (iters % self.save_interval == 0 or iters == num_iterations - 1):
            self.params['trainer'].save_checkpoint()
            LOGGER.text(f'Save model at [{iters}|{num_iterations}] to last.pth', LoggerObserver.INFO)


    def on_val_batch_end(self, logs:Dict=None):
        """
        On validation batch (iteration) end
        """

        iters = logs['iters']
        num_iterations = logs['num_iterations']
        metric_dict = logs['metric_dict']

        if self.best_key is None:
            return

        if not self.best_key in metric_dict.keys():
            LOGGER.text(
                f"{self.best_key} key does not present in metric. Available keys are: {metric_dict.keys()}", 
                LoggerObserver.WARN)
            return

        # Saving checkpoint
        if metric_dict[self.best_key] > self.best_value:
            if iters > 0: # Have been training, else in evaluation-only mode or just sanity check
                LOGGER.text(
                    f"Evaluation improved from {self.best_value} to {metric_dict[self.best_key]}",
                    level=LoggerObserver.INFO)
                self.best_value = metric_dict[self.best_key]
                self.params['trainer'].save_checkpoint('best')
                LOGGER.text(f'Save model at [{iters}|{num_iterations}] to best.pth', LoggerObserver.INFO)