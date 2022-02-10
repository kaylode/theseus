from distutils.log import Log
import os
from torckay.utilities.loggers.cp_logger import Checkpoint
from torckay.base.optimizers.scalers import NativeScaler

from torckay.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

class BaseTrainer():
    def __init__(self,
                model, 
                trainloader, 
                valloader,
                metrics,
                optimizer,
                scheduler,
                save_dir='runs',
                use_fp16=False, 
                num_epochs=100,
                total_accumulate_steps=None,
                clip_grad = 10.0,
                print_per_iter=100,
                save_per_iter=100,
                evaluate_per_epoch = 1,
                visualize_when_val = True,
                best_value = 0.0,
                resume = None,
                ):

        self.model = model
        self.metrics = metrics 
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainloader = trainloader
        self.valloader = valloader

        self.save_dir = save_dir
        self.checkpoint = Checkpoint(os.path.join(self.save_dir, 'checkpoints'))
        self.num_epochs = num_epochs
        self.step_per_epoch = self.scheduler.step_per_epoch
        self.use_amp = True if use_fp16 else False
        self.scaler = NativeScaler() if use_fp16 else False

        if total_accumulate_steps is None:
            self.accumulate_steps = 1
        else:
            self.accumulate_steps = max(round(total_accumulate_steps / trainloader.batch_size), 1) 
        self.clip_grad = clip_grad
        self.evaluate_per_epoch = evaluate_per_epoch
        self.print_per_iter = print_per_iter
        self.save_per_iter = save_per_iter
        self.visualize_when_val = visualize_when_val
        self.best_value = best_value
        self.resume = resume
        self.epoch = 0
        self.iters = 0
        self.start_iter = 0
        
    def fit(self):
        self.num_iters = (self.num_epochs+1) * len(self.trainloader)
        
        self.on_start()

        if self.step_per_epoch:
            self.scheduler.last_epoch = self.epoch - 1

        LOGGER.text(f'===========================START TRAINING=================================', level=LoggerObserver.INFO)
        for epoch in range(self.epoch, self.num_epochs):
            try:
                self.epoch = epoch
                self.training_epoch()
                self.on_training_end()

                if self.evaluate_per_epoch != 0:
                    if epoch % self.evaluate_per_epoch == 0 and epoch+1 >= self.evaluate_per_epoch:
                        self.evaluate_epoch()
                    self.on_evaluate_end()

                self.on_epoch_end()

            except KeyboardInterrupt:   
                break

        self.on_finish()
        LOGGER.text("Training Completed!", level=LoggerObserver.INFO)

    def sanity_check(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError
        
    def visualize_batch(self):
        raise NotImplementedError

    def training_epoch(self):
        raise NotImplementedError
    
    def evaluate_epoch(self):
        raise NotImplementedError
    
    def on_start(self):
        return

    def on_training_end(self):
        return

    def on_evaluate_end(self):
        return

    def on_epoch_end(self):
        if self.step_per_epoch:
            self.scheduler.step()
            lrl = [x['lr'] for x in self.optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            LOGGER.log([{
                'tag': 'Training/Learning rate',
                'value': lr,
                'type': LoggerObserver.SCALAR,
                'kwargs': {
                    'step': self.epoch
                }
            }])

    def on_finish(self):
        self.save_checkpoint()