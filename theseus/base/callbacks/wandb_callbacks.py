from typing import List, Dict
import os
from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.utilities.loggers.observer import LoggerObserver
from theseus.utilities.loggers.wandb_logger import WandbLogger
from datetime import datetime

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

    ::Usage::
    Register in the pipeline.yaml. For instance:

    callbacks:
    - name: WandbCallbacks
        args: 
        username: kaylode
        project_name: theseus

    """

    def __init__(self, 
        username: str, 
        project_name: str, 
        save_dir: str = None,
        **kwargs) -> None:
        super().__init__()

        self.username = username
        self.project_name = project_name

        # A hack, not good
        if save_dir is None:
            run_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_name = run_name

        """
        All the logging stuffs have been done in LoggerCallbacks. Here we just register 
        the wandb logger to the main logger
        """
        wandb_logger = WandbLogger(
            self.username, self.project_name, self.run_name
        )
        LOGGER.subscribe(wandb_logger)