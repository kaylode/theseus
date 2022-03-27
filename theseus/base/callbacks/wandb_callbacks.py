from typing import List, Dict
import os
import os.path as osp
from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.utilities.loggers.observer import LoggerObserver
from theseus.utilities.loggers.wandb_logger import WandbLogger, find_run_id
from datetime import datetime

try:
    import wandb as wandblogger
except ModuleNotFoundError:
    pass

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

    def __init__(self, 
        username: str, 
        project_name: str, 
        save_dir: str = None,
        resume: str = None,
        config_dict: Dict = None,
        **kwargs) -> None:
        super().__init__()

        self.username = username
        self.project_name = project_name
        self.resume = resume
        self.save_dir = save_dir
        self.config_dict = config_dict

        # A hack, not good
        if self.save_dir is None:
            self.run_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_name = osp.basename(save_dir)

        if self.resume is None:
            self.id = wandblogger.util.generate_id()
        else:
            self.id = find_run_id(
                os.path.dirname(os.path.dirname(self.resume))
            )

        """
        All the logging stuffs have been done in LoggerCallbacks. Here we just register 
        the wandb logger to the main logger
        """
        wandb_logger = WandbLogger(
            id = self.id,
            save_dir = self.save_dir,
            username = self.username, 
            project_name = self.project_name, 
            run_name = self.run_name,
            config_dict=self.config_dict
        )
        LOGGER.subscribe(wandb_logger)

    def on_start(self, logs: Dict=None):
        """
        Before going to the main loop. Save run id
        """
        wandb_id_file = osp.join(self.save_dir, 'wandb_id.txt')
        with open(wandb_id_file, 'w') as f:
            f.write(self.id)
        