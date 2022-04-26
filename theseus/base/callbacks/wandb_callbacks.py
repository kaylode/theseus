from typing import Dict
import os
import os.path as osp
from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.utilities.loggers.observer import LoggerObserver
from theseus.utilities.loggers.wandb_logger import WandbLogger, find_run_id
from datetime import datetime
from theseus.opt import Config
from copy import deepcopy

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
            try:
                # Get run id
                run_id = find_run_id(
                    os.path.dirname(os.path.dirname(self.resume))
                )

                # Load the config from that run
                try:
                    old_config_path = wandblogger.restore(
                        'pipeline.yaml',
                        run_path = f"{self.username}/{self.project_name}/{run_id}"
                    ).name
                except Exception:
                    raise ValueError(f"Falid to load run id={run_id}, due to pipeline.yaml is missing or run is not existed")

                # Check if the config remains the same, if not, create new run id 
                old_config_dict = Config(old_config_path)
                tmp_config_dict = deepcopy(self.config_dict)
                ## strip off global key because `resume` will always different
                old_config_dict.pop('global', None)
                tmp_config_dict.pop('global', None)
                if old_config_dict == tmp_config_dict:
                    self.id = run_id
                    LOGGER.text("Run configuration remains unchanged. Resuming wandb run...", LoggerObserver.SUCCESS)
                else:
                    self.id = wandblogger.util.generate_id()
                    LOGGER.text("Run configuration changes since the last run. Creating new wandb run...", LoggerObserver.WARN)
            except ValueError as e:
                LOGGER.text(f"Can not resume wandb due to '{e}'. Creating new wandb run...", LoggerObserver.WARN)
                self.id = wandblogger.util.generate_id()


        # All the logging stuffs have been done in LoggerCallbacks. 
        # Here we just register the wandb logger to the main logger

        self.wandb_logger = WandbLogger(
            unique_id = self.id,
            save_dir = self.save_dir,
            username = self.username, 
            project_name = self.project_name, 
            run_name = self.run_name,
            config_dict=self.config_dict
        )
        LOGGER.subscribe(self.wandb_logger)

    def on_start(self, logs: Dict=None):
        """
        Before going to the main loop. Save run id
        """
        wandb_id_file = osp.join(self.save_dir, 'wandb_id.txt')
        with open(wandb_id_file, 'w') as f:
            f.write(self.id)

        # Save all config files
        self.wandb_logger.log_file(
            tag='configs', 
            value = osp.join(self.save_dir, '*.yaml'))
        
        # Init logging model for debug
        self.wandb_logger.log_torch_module(
            tag='models', 
            value = self.params['trainer'].model.model,
            log_freq=10)

    def on_finish(self, logs: Dict=None):
        """
        After finish training
        """
        base_folder=osp.join(self.save_dir, 'checkpoints')
        self.wandb_logger.log_file(
            tag='checkpoint', 
            base_folder=self.save_dir,
            value = osp.join(base_folder, '*.pth'))

    def on_val_epoch_end(self, logs:Dict=None):
        """
        On validation batch (iteration) end
        """ 
        base_folder=osp.join(self.save_dir, 'checkpoints')
        self.wandb_logger.log_file(
            tag='checkpoint', 
            base_folder=self.save_dir,
            value = osp.join(base_folder, '*.pth'))