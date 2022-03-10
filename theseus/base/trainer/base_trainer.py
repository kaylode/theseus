from typing import List, Optional, Tuple

import os
from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.base.optimizers.scalers import NativeScaler
from theseus.base.callbacks import CallbacksList, LoggerCallbacks
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
                callbacks: List[Callbacks] = [LoggerCallbacks()]
                ):

        self.save_dir = save_dir
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
        self.callbacks = callbacks if isinstance(callbacks, list) else [callbacks]
        self.callbacks = CallbacksList(self.callbacks)
        self.callbacks.set_params({'trainer': self})
        
    def fit(self): 
        
        # On start callbacks
        self.callbacks.run('on_start')

        while self.iters < self.num_iterations:
            try:
                # On epoch start callbacks
                self.callbacks.run('on_epoch_start', {'iters': self.iters})

                # Start training 
                self.training_epoch()

                # Start evaluation 
                if self.evaluate_interval != 0:
                    if self.iters % self.evaluate_interval == 0 and self.iters>0:
                        self.evaluate_epoch()

                # On epoch end callbacks
                self.callbacks.run('on_epoch_end', {'iters': self.iters})

            except KeyboardInterrupt:   
                break
        
        # On finish callbacks
        self.callbacks.run('on_finish')