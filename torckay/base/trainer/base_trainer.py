import os
from torckay.utilities.loggers.logger import LoggerManager
from torckay.utilities.loggers.cp_logger import Checkpoint
from torckay.utilities.loggers.tf_logger import TensorboardLogger

LOGGER = LoggerManager.init_logger(__name__)

class BaseTrainer():
    def __init__(self,
                model, 
                trainloader, 
                valloader,
                metrics,
                optimizer,
                scheduler,
                save_dir='runs',
                scaler=None, 
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
        self.tf_logger = TensorboardLogger(self.save_dir)
        self.step_per_epoch = self.scheduler.step_per_epoch
        self.scaler = scaler
        self.use_amp = True if scaler is not None else False
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
        
        self.on_training_start()

        if self.step_per_epoch:
            self.scheduler.last_epoch = self.epoch - 1

        LOGGER.info(f'===========================START TRAINING=================================')
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
        LOGGER.info("Training Completed!")

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

    def on_evaluate_end(self):
        if self.visualize_when_val:
            self.visualize_batch()
        self.save_checkpoint()
    
    def on_training_start(self):
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
            log_dict = {'Training/Learning rate': lr}
            self.tf_logger.write_dict(log_dict, step=self.epoch)

    def on_finish(self):
        self.save_checkpoint()