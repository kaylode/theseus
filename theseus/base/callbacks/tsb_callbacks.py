import os
from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.utilities.loggers.observer import LoggerObserver
from theseus.utilities.loggers.tsb_logger import TensorboardLogger
from theseus.utilities.loading import find_old_tflog

LOGGER = LoggerObserver.getLogger("main")

class TensorboardCallbacks(Callbacks):
    """
    Callbacks for logging running loss/metric/time while training to tensorboard
    Features:
        - Only do logging
    
    save_dir: `str`
        directory to save checkpoints
    resume: `str`
        iteration cycle to log out
    """

    def __init__(self, save_dir: str = 'runs', resume: str = 10, **kwargs) -> None:
        super().__init__()

        self.save_dir = save_dir
        self.resume = resume

        """
        All the logging stuffs have been done in LoggerCallbacks. Here we just register 
        the tensorboard logger to the main logger
        """
        tsb_logger = TensorboardLogger(self.save_dir)
        if self.resume is not None:
            tsb_logger.load(find_old_tflog(
                os.path.dirname(os.path.dirname(self.resume))
            ))
        LOGGER.subscribe(tsb_logger)