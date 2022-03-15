from typing import List, Dict
from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.utilities.loggers.observer import LoggerObserver
import time
import numpy as np

LOGGER = LoggerObserver.getLogger("main")

class LoggerCallbacks(Callbacks):
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
        self.running_loss = {}
        self.print_interval = print_interval

    def sanitycheck(self, logs: Dict=None):
        """
        Sanitycheck before starting. Run only when debug=True
        """
        LOGGER.text("Start sanity checks", level=LoggerObserver.DEBUG)

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

            # Start new interval
            self.running_loss = {}
            self.running_time = time.time()

    def on_val_epoch_start(self, logs: Dict=None):
        """
        Before main validation loops
        """
        LOGGER.text('=============================EVALUATION===================================', LoggerObserver.INFO)
        self.running_time = time.time()
        self.running_loss = {}

    def on_val_batch_end(self, logs: Dict=None):
        """
        After finish a batch
        """

        loss_dict = logs['loss_dict']

        # Update batch loss
        for (key,value) in loss_dict.items():
            if key in self.running_loss.keys():
                self.running_loss[key] += value
            else:
                self.running_loss[key] = value

    def on_val_epoch_end(self, logs: Dict=None):
        """
        After finish validation
        """

        iters = logs['iters']
        metric_dict = logs['metric_dict']
        num_iterations = logs['num_iterations']
        epoch_time = time.time() - self.running_time
        valloader = self.params['trainer'].valloader

        # Log loss
        for key in self.running_loss.keys():
            self.running_loss[key] /= len(valloader)
            self.running_loss[key] = np.round(self.running_loss[key], 5)
        loss_string = '{}'.format(self.running_loss)[1:-1].replace("'",'').replace(",",' ||')
        LOGGER.text(
            "[{}|{}] || {} || Time: {:10.4f} (it/s)".format(
                iters, num_iterations, loss_string, len(valloader)/epoch_time),
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
            'tag': f"Validation/{k} Loss",
            'value': v/len(valloader),
            'type': LoggerObserver.SCALAR,
            'kwargs': {
                'step': iters
            }
        } for k,v in self.running_loss.items()]

        log_dict += [{
            'tag': f"Validation/{k}",
            'value': v,
            'kwargs': {
                'step': iters
            }
        } for k,v in metric_dict.items()]

        LOGGER.log(log_dict)