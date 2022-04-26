import torch
import os

from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

class Checkpoint():
    """
    Checkpoint for saving state dict
    :param save_per_epoch: (int)
    :param path: (string)
    """
    def __init__(self, path):
        self.path = path
        
    def save(self, state_dict, outname):
        """
        Save model and optimizer weights
        :param model: Pytorch model with state dict
        """
        os.makedirs(self.path, exist_ok=True)
        torch.save(state_dict, os.path.join(self.path,outname)+".pth")
        LOGGER.text(
            f"Save checkpoints to {os.path.join(self.path,outname)}"+".pth",
            level=LoggerObserver.INFO)
