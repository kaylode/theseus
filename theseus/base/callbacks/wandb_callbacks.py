from typing import List, Dict
import os
from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.utilities.loggers.observer import LoggerObserver
from theseus.utilities.loggers.wandb_logger import WandbLogger

LOGGER = LoggerObserver.getLogger("main")

class WandbCallbacks(Callbacks):
    """
    Callbacks for logging running loss/metric/time while training to wandb server
    Features:
        - Only do logging
    
    username: `str`
        username of Wandb
    project_name: `str`
        project name of Wandb
    resume: `bool`
        whether to resume project
    """

    def __init__(self, username: str, project_name: str, resume: bool = False, **kwargs) -> None:
        super().__init__()

        self.username = username
        self.project_name = project_name
        self.resume = resume

        """
        All the logging stuffs have been done in LoggerCallbacks. Here we just register 
        the wandb logger to the main logger
        """
        wandb_logger = WandbLogger(
            self.username, self.project_name, self.resume
        )
        LOGGER.subscribe(wandb_logger)