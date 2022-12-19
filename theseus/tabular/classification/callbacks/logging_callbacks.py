from typing import List, Dict
from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.base.utilities.loggers.observer import LoggerObserver
import time

LOGGER = LoggerObserver.getLogger("main")

class MLLoggerCallbacks(Callbacks):
    """
    Callbacks for logging running loss/metric/time while training
    Features:
        - Only do logging
        
    print_interval: `int`
        iteration cycle to log out
    """

    def __init__(self, print_interval: int = 10, **kwargs) -> None:
        super().__init__()

        self.running_time = 0
        self.print_interval = print_interval

    def on_start(self, logs: Dict=None):
        """
        Before going to the main loop
        """
        LOGGER.text(f'===========================START TRAINING=================================', level=LoggerObserver.INFO)

    def on_finish(self, logs: Dict=None):
        """
        After the main loop
        """
        LOGGER.text("Training Completed!", level=LoggerObserver.INFO)

    def on_train_epoch_start(self, logs: Dict=None):
        """
        Before going to the training loop
        """
        self.running_time = time.time()

    def on_train_epoch_end(self, logs: Dict=None):
        """
        Before going to the training loop
        """
        epoch_time = time.time() - self.running_time
        LOGGER.text(
            "Total training time: {:10.4f} seconds".format(epoch_time),
        level=LoggerObserver.INFO)

    def on_val_epoch_start(self, logs: Dict=None):
        """
        Before main validation loops
        """
        LOGGER.text('=============================EVALUATION===================================', LoggerObserver.INFO)
        self.running_time = time.time()

    def on_val_epoch_end(self, logs: Dict=None):
        """
        After finish validation
        """

        iters = logs['iters']
        metric_dict = logs['metric_dict']
        epoch_time = time.time() - self.running_time

        LOGGER.text(
            "Total evaluation time: {:10.4f} seconds".format(epoch_time),
        level=LoggerObserver.INFO)

        # Log metric
        metric_string = ""
        for metric, score in metric_dict.items():
            if isinstance(score, (int, float)):
                metric_string += metric +': ' + f"{score:.5f}" +' | '
        metric_string +='\n'

        LOGGER.text(metric_string, level=LoggerObserver.INFO)
        LOGGER.text('==========================================================================', level=LoggerObserver.INFO)

        # Call other loggers
        log_dict = [{
            'tag': f"Validation/{k}",
            'value': v,
            'kwargs': {
                'step': iters
            }
        } for k,v in metric_dict.items()]

        LOGGER.log(log_dict)