from typing import List, Optional, Tuple

import os
from theseus.utilities.loggers.cp_logger import Checkpoint
from theseus.base.optimizers.scalers import NativeScaler
from theseus.base.callbacks import CallbacksList, DefaultCallbacks
from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

class BaseTrainer():
    """Base class for trainer

    save_dir: `str`
        Path to directory for saving stuffs
    use_fp16: `bool`
        whether to use 16bit floating-point precision 
    num_iterations: `int`
        total number of running epochs
    total_accumulate_steps: `int`
        gradient accumulation step. None means not use
    clip_grad: `float`
        Gradient clipping
    print_interval: `int`
        Logging cycle per iteration
    save_interval: `int`
        Save checkpoint per iteration
    evaluate_interval: `int`
        Number of epochs to perform validation
    visualize_when_val: `bool`
        whether to visualize predictions
    best_value: `float`
        Current best value of evaluation metric
    resume: `str`
        Path to checkpoint for continue training
    """
    def __init__(self,
                save_dir: str = 'runs',
                use_fp16: bool = False, 
                num_iterations: int = 10000,
                clip_grad: float = 10.0,
                print_interval: int = 100,
                save_interval: int = 100,
                evaluate_interval: int = 1,
                visualize_when_val: bool = True,
                best_value: float = 0.0,
                resume: str = Optional[None],
                callbacks: CallbacksList = CallbacksList([DefaultCallbacks()])
                ):

        self.save_dir = save_dir
        self.checkpoint = Checkpoint(os.path.join(self.save_dir, 'checkpoints'))
        self.num_iterations = num_iterations
        self.use_amp = True if use_fp16 else False
        self.scaler = NativeScaler() if use_fp16 else False
        self.clip_grad = clip_grad
        self.evaluate_interval = evaluate_interval
        self.print_interval = print_interval
        self.save_interval = save_interval
        self.visualize_when_val = visualize_when_val
        self.best_value = best_value
        self.resume = resume
        self.iters = 0
        self.callbacks = callbacks
        self.callbacks.set_params(self)
        
    def fit(self): 
        
        # On start callbacks
        self.callbacks.run('on_start')

        while self.iters < self.num_iterations:
            try:
                # Start training
                self.training_epoch()
                self.on_training_end()

                # Start evaluation
                if self.evaluate_interval != 0:
                    if self.iters % self.evaluate_interval == 0 and self.iters>0:
                        self.evaluate_epoch()
                    self.on_evaluate_end()
                
                # On epoch end callbacks
                self.on_epoch_end()

            except KeyboardInterrupt:   
                break
        
        # On training finish callbacks
        
        self.callbacks.run('on_finish')


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
        return

    def on_finish(self):
        return