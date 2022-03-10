from typing import List, Dict
import numpy as np
from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

class SupervisedCallbacks:
    """
    Default callbacks that will always be used
    """
    def __init__(self) -> None:
        pass

    def on_train_epoch_start(self, logs: Dict=None):
        """
        On every training epoch starts
        """
        trainer = logs['trainer']
        trainer.model.train()
        trainer.optimizer.zero_grad()

    def on_train_epoch_end(self, logs: Dict=None):
        """
        After the main loop
        """

    def optimizer_step(self, logs:Dict=None):
        """
        Perform loss backward and optimizer step, scheduler step
        """

        trainer = logs['trainer']
        loss = logs['loss']

        # Backward loss
        self.scaler(loss, trainer.optimizer)
        
        # Optmizer step
        trainer.scaler.step(
            trainer.optimizer, 
            clip_grad=trainer.clip_grad, 
            parameters=trainer.model.parameters())

        if not trainer.step_per_epoch:
            trainer.scheduler.step()
            lrl = [x['lr'] for x in self.optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            LOGGER.log([{
                'tag': 'Training/Learning rate',
                'value': lr,
                'type': LoggerObserver.SCALAR,
                'kwargs': {
                    'step': self.iters
                }
            }])

        trainer.optimizer.zero_grad()
        

    def on_train_batch_end(self, logs:Dict=None):
        """
        On training batch (iteration) end
        """

        trainer = logs['trainer']
        loss_dict = logs['loss_dict']
        start_time = logs['start_time']
        end_time = logs['end_time']

        for (key,value) in loss_dict.items():
            if key in running_loss.keys():
                running_loss[key] += value
            else:
                running_loss[key] = value

        # Calculate current iteration
        trainer.iters = trainer.iters + 1

        # Logging
        if trainer.iters % trainer.print_interval == 0:
            for key in running_loss.keys():
                running_loss[key] /= trainer.print_interval
                running_loss[key] = np.round(running_loss[key], 5)
            loss_string = '{}'.format(running_loss)[1:-1].replace("'",'').replace(",",' ||')

            LOGGER.text(
                "[{}|{}] || {} || Time: {:10.4f} (it/s)".format(
                    trainer.iters, trainer.num_iterations,
                    loss_string, trainer.print_interval/running_time), 
                LoggerObserver.INFO)
            
            log_dict = [{
                'tag': f"Training/{k} Loss",
                'value': v/trainer.print_interval,
                'type': LoggerObserver.SCALAR,
                'kwargs': {
                    'step': trainer.iters
                }
            } for k,v in running_loss.items()]


            log_dict.append({
                'tag': f"Training/Iterations per second",
                'value': trainer.print_interval/running_time,
                'type': LoggerObserver.SCALAR,
                'kwargs': {
                    'step': trainer.iters
                }
            })
            LOGGER.log(log_dict)

            running_loss = {}
            running_time = 0

        # Saving checkpoint
        if (self.iters % self.save_interval == 0 or self.iters == self.num_iterations - 1):
            LOGGER.text(f'Save model at [{self.iters}|{self.num_iterations}] to last.pth', LoggerObserver.INFO)
            self.save_checkpoint()