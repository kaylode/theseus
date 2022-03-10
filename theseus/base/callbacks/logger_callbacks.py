from typing import List, Dict
from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.utilities.loggers.observer import LoggerObserver
import time
import numpy as np

LOGGER = LoggerObserver.getLogger("main")

class LoggerCallbacks(Callbacks):
    """
    Default callbacks that will always be used
    """

    def __init__(self, print_interval: int = 10) -> None:
        super().__init__()

        self.running_time = 0
        self.running_loss = {}
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
        self.running_loss = {}
        self.running_time = 0

    def on_train_batch_start(self, logs: Dict=None):
        """
        Before beginning a batch
        """
        self.running_time = time.time()

    def on_train_batch_end(self, logs: Dict=None):
        """
        After finish a batch
        """

        lr = logs['lr']
        iters = logs['iters']
        loss_dict = logs['loss_dict']
        num_iterations = logs['num_iterations']

        # Update running loss of batch
        for (key,value) in loss_dict.items():
            if key in self.running_loss.keys():
                self.running_loss[key] += value
            else:
                self.running_loss[key] = value

        # Logging
        if iters % self.print_interval == 0:

            # Running time since last interval
            batch_time = time.time() - self.running_time

            # Running loss since last interval
            for key in self.running_loss.keys():
                self.running_loss[key] /= self.print_interval
                self.running_loss[key] = np.round(self.running_loss[key], 5)
            loss_string = '{}'.format(self.running_loss)[1:-1].replace("'",'').replace(",",' ||')

            LOGGER.text(
                "[{}|{}] || {} || Time: {:10.4f} (it/s)".format(
                    iters, num_iterations,
                    loss_string, self.print_interval/batch_time), 
                LoggerObserver.INFO)
            
            log_dict = [{
                'tag': f"Training/{k} Loss",
                'value': v/self.print_interval,
                'type': LoggerObserver.SCALAR,
                'kwargs': {
                    'step': iters
                }
            } for k,v in self.running_loss.items()]


            # Log batch time execution
            log_dict.append({
                'tag': f"Training/Iterations per second",
                'value': self.print_interval/batch_time,
                'type': LoggerObserver.SCALAR,
                'kwargs': {
                    'step': iters
                }
            })

            # Log learning rates
            log_dict.append({
                'tag': 'Training/Learning rate',
                'value': lr,
                'type': LoggerObserver.SCALAR,
                'kwargs': {
                    'step': iters
                }
            })

            
            LOGGER.log(log_dict)
            self.running_loss = {}
            self.running_time = time.time()