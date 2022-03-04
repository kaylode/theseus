try:
    import theseus.utilities.loggers.wandb_logger as wandb_logger
except ModuleNotFoundError:
    pass
import torch
from theseus.utilities.loggers.observer import LoggerObserver, LoggerSubscriber
LOGGER = LoggerObserver.getLogger('main')

class WandbLogger(LoggerSubscriber):
    """
    Logger for wandb intergration
    :param log_dir: Path to save checkpoint
    """
    def __init__(self, username:str, project_name:str, resume:bool = False):
        self.project_name = project_name
        self.username = username
        self.resume = resume
        
        wandb_logger.init(entity=username, project=project_name, resume=resume)
        wandb_logger.watch_called = False

    def load_state_dict(self, path):
        if wandb_logger.run.resumed:
            state_dict = torch.load(wandb_logger.restore(path))
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

        wandb_logger.log({tag: value}, step=step)

    def log_figure(self, tag, value, step, **kwargs):
        """
        Write a matplotlib fig to tensorboard
        :param tags: (str) tag for log
        :param value: (image) image to log. torch.Tensor or plt.fire.Figure
        :param step: (int) logging step
        """


        if isinstance(value, torch.Tensor):
            image = wandb_logger.Image(value)
            wandb_logger.log({
               tag: image
            }, step=step)
        else:
            wandb_logger.log({
               tag: value
            }, step=step)

    def log_torch_module(self, tag, value, **kwargs):
        """
        Write a model graph to tensorboard
        :param value: (nn.Module) torch model
        :param inputs: sample tensor
        """
        wandb_logger.watch(value, log="all")

    def __del__(self):
        wandb_logger.finish()