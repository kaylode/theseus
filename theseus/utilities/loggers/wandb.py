import wandb
import torch
from theseus.utilities.loggers.observer import LoggerObserver, LoggerSubscriber
LOGGER = LoggerObserver.getLogger('main')

class WandbLogger(LoggerSubscriber):
    """
    Logger for wandb intergration
    :param log_dir: Path to save checkpoint
    """
    def __init__(self, name:str, resume:bool = False):
        self.name = name
        self.resume = resume
        
        wandb.init(entity="wandb", project=name, resume=resume)
        wandb.watch_called = False

    def load_state_dict(self, path):
        if wandb.run.resumed:
            state_dict = torch.load(wandb.restore(path))
            return state_dict
        else:
            return None

    def log_scalar(self, tag, value, step, **kwargs):
        """
        Write a log to specified directory
        :param tags: (str) tag for log
        :param values: (number) value for corresponding tag
        :param step: (int) logging step
        """

        wandb.log({tag: value}, step=step)

    def log_figure(self, tag, value, step, **kwargs):
        """
        Write a matplotlib fig to tensorboard
        :param tags: (str) tag for log
        :param value: (image) image to log. torch.Tensor or plt.fire.Figure
        :param step: (int) logging step
        """


        if isinstance(value, torch.Tensor):
            image = wandb.Image(value)
            wandb.log({
               tag: image
            }, step=step)
        else:
            wandb.log({
               tag: value
            }, step=step)

    def log_torch_module(self, tag, value, **kwargs):
        """
        Write a model graph to tensorboard
        :param value: (nn.Module) torch model
        :param inputs: sample tensor
        """
        wandb.watch(value, log="all")

    def __del__(self):
        wandb.finish()