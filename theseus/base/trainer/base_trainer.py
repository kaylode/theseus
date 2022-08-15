from typing import List, Optional, Tuple

from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.base.optimizers.scalers import NativeScaler
from theseus.base.callbacks import CallbacksList, LoggerCallbacks, CheckpointCallbacks
from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

class BaseTrainer():
    """Base class for trainer

    use_fp16: `bool`
        whether to use 16bit floating-point precision 
    num_iterations: `int`
        total number of running epochs
    clip_grad: `float`
        Gradient clipping
    evaluate_interval: `int`
        Number of epochs to perform validation
    resume: `str`
        Path to checkpoint for continue training
    """
    def __init__(self,
                use_fp16: bool = False, 
                num_iterations: int = 10000,
                clip_grad: float = 10.0,
                evaluate_interval: int = 1,
                callbacks: List[Callbacks] = [LoggerCallbacks(), CheckpointCallbacks()],
                debug: bool = False,
                **kwargs
                ):

        self.num_iterations = num_iterations
        self.use_amp = True if use_fp16 else False
        self.scaler = NativeScaler(use_fp16)
        self.clip_grad = clip_grad
        self.evaluate_interval = evaluate_interval
        self.iters = 0
        self.debug = debug
        self.shutdown_all = False # Flag to stop trainer imediately

        if not isinstance(callbacks, CallbacksList):
            callbacks = callbacks if isinstance(callbacks, list) else [callbacks]
            callbacks = CallbacksList(callbacks)
        self.callbacks = callbacks
        self.callbacks.set_params({'trainer': self})
        
    def fit(self): 
        
        # Sanity check if debug is set
        if self.debug:
            self.callbacks.run('sanitycheck', {
                'iters': self.iters,
                'num_iterations': self.num_iterations
            })

        # On start callbacks
        self.callbacks.run('on_start')

        while self.iters < self.num_iterations:
            try:

                # Check if shutdown flag has been turned on
                if self.shutdown_all:
                    break

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
        self.callbacks.run('on_finish', {
            'iters': self.iters,
            'num_iterations': self.num_iterations
        })