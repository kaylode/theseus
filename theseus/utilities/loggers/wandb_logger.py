from typing import Dict
try:
    import wandb as wandb_logger
except ModuleNotFoundError:
    pass

import os.path as osp
import torch
from theseus.utilities.loggers.observer import LoggerObserver, LoggerSubscriber
LOGGER = LoggerObserver.getLogger('main')

class WandbLogger(LoggerSubscriber):
    """
    Logger for wandb intergration
    :param log_dir: Path to save checkpoint
    """
    def __init__(self, unique_id:str, username:str, project_name:str, run_name:str, save_dir:str = None, config_dict:Dict = None):
        self.project_name = project_name
        self.username = username
        self.run_name = run_name
        self.config_dict = config_dict
        self.id = unique_id
        self.save_dir = save_dir
        
        wandb_logger.init(
            id = self.id,
            dir = self.save_dir,
            config=config_dict,
            entity=username, 
            project=project_name, 
            name=run_name, 
            resume="allow")
            
        wandb_logger.watch_called = False

    def load_state_dict(self, path):
        if wandb_logger.run.resumed:
            state_dict = torch.load(wandb_logger.restore(path))
            return state_dict
        else:
            return None

    def log_file(self, tag, value, base_folder=None, **kwargs):
        """
        Write a file to wandb
        :param tag: (str) tag
        :param value: (str) path to file

        :param base_folder: (str) folder to save file to
        """
        wandb_logger.save(value, base_path=base_folder)


    def log_scalar(self, tag, value, step, **kwargs):
        """
        Write a log to specified directory
        :param tags: (str) tag for log
        :param values: (number) value for corresponding tag
        :param step: (int) logging step
        """

        wandb_logger.log({
            tag: value,
            'iterations': step
        })

    def log_figure(self, tag, value, step, **kwargs):
        """
        Write a matplotlib fig to wandb
        :param tags: (str) tag for log
        :param value: (image) image to log. torch.Tensor or plt.fire.Figure
        :param step: (int) logging step
        """

        if isinstance(value, torch.Tensor):
            image = wandb_logger.Image(value)
            wandb_logger.log({
               tag: image,
               'iterations': step
            })
        else:
            wandb_logger.log({
               tag: value,
               'iterations': step
            })

    def log_torch_module(self, tag, value, log_freq, **kwargs):
        """
        Write a model graph to wandb
        :param value: (nn.Module) torch model
        :param inputs: sample tensor
        """
        wandb_logger.watch(
          value, 
          log="gradients", 
          log_freq=log_freq)

    def log_spec_text(self, tag, value, step, **kwargs):
        """
        Write a text to wandb
        :param value: (str) captions
        """
        texts = wandb_logger.Html(value)
        wandb_logger.log({
            tag: texts,
            'iterations': step
        })

    def log_table(self, tag, value, columns, step, **kwargs):
        """
        Write a table to wandb
        :param value: list of column values
        :param columns: list of column names

        Examples:
        value = [
            [0, fig1, 0],
            [1, fig2, 8],
            [2, fig3, 7],
            [3, fig4, 1]
        ]
        columns=[
            "id", 
            "image", 
            "prediction"
        ]
        """
        
        # Workaround for tensor image, have not figured out how to use plt.Figure :<
        new_value = []
        for record in value:
            new_record = []
            for val in record:
                if isinstance(val, torch.Tensor):
                    val = wandb_logger.Image(val)
                new_record.append(val)
            new_value.append(new_record)

        table = wandb_logger.Table(data=new_value, columns=columns)
        wandb_logger.log({
            tag: table,
            'iterations': step
        })

    def log_video(self, tag, value, step, fps, **kwargs):
        """
        Write a video to wandb
        :param value: numpy array (time, channel, height, width)
        :param fps: int
        """
        # axes are 
        wandb_logger.log({
            tag: wandb_logger.Video(value, fps=fps),
            "iterations": step
        })

    def __del__(self):
        wandb_logger.finish()


def find_run_id(dirname):
    """
    Read a .txt file which contains wandb run id
    """

    wandb_id_file = osp.join(dirname, 'wandb_id.txt')

    if not osp.isfile(wandb_id_file):
        raise ValueError(f"Wandb ID file not found in {wandb_id_file}")
    else:
        with open(wandb_id_file, 'r') as f:
            wandb_id = f.read().rstrip()
        return wandb_id