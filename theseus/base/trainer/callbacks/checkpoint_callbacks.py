from distutils.log import WARN
from typing import List, Dict
import os.path as osp
import torch
from theseus.utilities.loading import load_state_dict
from theseus.base.trainer.callbacks import Callbacks
from theseus.utilities.loggers.cp_logger import Checkpoint
from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

class CheckpointCallbacks(Callbacks):
    """
    Callbacks for saving checkpoints.

    Features:
        - Load checkpoint at start
        - Save checkpoint every save_interval
        - Save checkpoint if metric value is improving
    """

    def __init__(self, save_dir: str='runs', save_interval: int=10, best_key:str = None) -> None:
        super().__init__()

        self.best_value = 0
        self.best_key = best_key
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.checkpoint = Checkpoint(osp.join(self.save_dir, 'checkpoints')) 

    def load_checkpoint(self, path, trainer):
        """
        Load all information the current iteration from checkpoint 
        """
        LOGGER.text("Loading checkpoints...", level=LoggerObserver.INFO)
        state_dict = torch.load(path, map_location='cpu')
        trainer.iters = load_state_dict(trainer.iters, state_dict, 'iters')
        trainer.scaler = load_state_dict(trainer.scaler, state_dict, trainer.scaler.state_dict_key)
        self.best_value = load_state_dict(self.best_value, state_dict, 'best_value')  

    def save_checkpoint(self, trainer, iters, outname='last'):
        """
        Save all information of the current iteration
        """
        weights = {
            'model': trainer.model.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            'iters': iters,
            'best_value': self.best_value,
        }

        if trainer.scaler is not None:
            weights[trainer.scaler.state_dict_key] = trainer.scaler.state_dict()
           
        self.checkpoint.save(weights, outname)

    def on_start(self, logs: Dict=None):
        """
        Before going to the main loop
        """
        resume = logs['resume']
        self.load_checkpoint(resume, self.params['trainer'])

    def on_train_batch_end(self, logs:Dict=None):
        """
        On training batch (iteration) end
        """

        iters = logs['iters']
        num_iterations = logs['num_iterations']

        # Saving checkpoint
        if (iters % self.save_interval == 0 or iters == num_iterations - 1):
            self.save_checkpoint(self.params['trainer'], iters=iters)
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
                self.save_checkpoint(self.params['trainer'], iters=iters, outname='best')

                LOGGER.text(f'Save model at [{iters}|{num_iterations}] to best.pth', LoggerObserver.INFO)