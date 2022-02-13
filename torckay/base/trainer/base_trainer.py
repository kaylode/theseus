from typing import List, Optional, Tuple

import os
from torckay.utilities.loggers.cp_logger import Checkpoint
from torckay.base.optimizers.scalers import NativeScaler

from torckay.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

class BaseTrainer():
    """Base class for trainer

    model : `torch.nn.Module`
        Wrapper model with loss 
    trainloader : `torch.utils.DataLoader`
        DataLoader for training
    valloader : `torch.utils.DataLoader`
        DataLoader for validation
    metrics: `List[Metric]`
        list of metrics for evaluation
    optimizer: `torch.optim.Optimizer`
        optimizer for parameters update
    scheduler: `torch.optim.lr_scheduler.Scheduler`
        learning rate schedulers
    save_dir: `str`
        Path to directory for saving stuffs
    use_fp16: `bool`
        whether to use 16bit floating-point precision 
    num_epochs: `int`
        total number of running epochs
    total_accumulate_steps: `int`
        gradient accumulation step. None means not use
    clip_grad: `float`
        Gradient clipping
    print_per_iter: `int`
        Logging cycle per iteration
    save_per_iter: `int`
        Save checkpoint per iteration
    evaluate_per_epoch: `int`
        Number of epochs to perform validation
    visualize_when_val: `bool`
        whether to visualize predictions
    best_value: `float`
        Current best value of evaluation metric
    resume: `str`
        Path to checkpoint for continue training
    """
    def __init__(self,
                model, 
                trainloader, 
                valloader,
                metrics,
                optimizer,
                scheduler,
                save_dir: str = 'runs',
                use_fp16: bool = False, 
                num_epochs: int = 100,
                total_accumulate_steps: Optional[int] = None,
                clip_grad: float = 10.0,
                print_per_iter: int = 100,
                save_per_iter: int = 100,
                evaluate_per_epoch: int = 1,
                visualize_when_val: bool = True,
                best_value: float = 0.0,
                resume: str = Optional[None],
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
        # Total number of training iterations
        self.num_iters = (self.num_epochs+1) * len(self.trainloader)
        
        # On start callbacks
        self.on_start()

        # Init scheduler params
        if self.step_per_epoch:
            self.scheduler.last_epoch = self.epoch - 1

        LOGGER.text(f'===========================START TRAINING=================================', level=LoggerObserver.INFO)
        for epoch in range(self.epoch, self.num_epochs):
            try:
                # Save current epoch
                self.epoch = epoch

                # Start training
                self.training_epoch()
                self.on_training_end()

                # Start evaluation
                if self.evaluate_per_epoch != 0:
                    if epoch % self.evaluate_per_epoch == 0 and epoch+1 >= self.evaluate_per_epoch:
                        self.evaluate_epoch()
                    self.on_evaluate_end()
                
                # On epoch end callbacks
                self.on_epoch_end()

            except KeyboardInterrupt:   
                break
        
        # On training finish callbacks
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