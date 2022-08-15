import math
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, LambdaLR, MultiStepLR,
    ReduceLROnPlateau,OneCycleLR, CosineAnnealingWarmRestarts)
from .cosine import CosineWithRestarts

class SchedulerWrapper():
    """
    Wrap scheduler, factory design pattern
    """
    def __init__(self, optimizer, scheduler_name,  **kwargs) -> None:

        if scheduler_name == '1cycle-yolo':
            def one_cycle(y1=0.0, y2=1.0, steps=100):
                # lambda function for sinusoidal ramp from y1 to y2
                return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

            lf = one_cycle(1, 0.2, kwargs['num_epochs'])  # cosine 1->hyp['lrf']
            scheduler = LambdaLR(optimizer, lr_lambda=lf)
            step_per_epoch = True
            
        elif scheduler_name == '1cycle':
            scheduler = OneCycleLR(
                optimizer,
                max_lr=0.001,
                epochs=kwargs['num_epochs'],
                steps_per_epoch=int(len(kwargs["trainset"]) / kwargs["batch_size"]),
                pct_start=0.1,
                anneal_strategy='cos', 
                final_div_factor=10**5)
            step_per_epoch = False

        elif scheduler_name == 'multistep':
            scheduler = MultiStepLR(
                optimizer,
                milestones=kwargs['milestones'],
                gamma=kwargs['gamma'], 
                last_epoch=kwargs['last_epoch'])
            step_per_epoch = False

        elif scheduler_name == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=1,
                verbose=False, 
                threshold=0.0001,
                threshold_mode='abs',
                cooldown=0, 
                min_lr=1e-8,
                eps=1e-08
            )
            step_per_epoch = True

        elif scheduler_name == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=kwargs['num_epochs'],
                T_mult=1,
                eta_min=0.0001,
                last_epoch=kwargs['last_epoch'],
                verbose=False
            )
            step_per_epoch = False

        elif scheduler_name == 'cosine2':
            scheduler = CosineWithRestarts(
                optimizer, 
                t_initial=kwargs['t_initial'],
                t_mul=kwargs['t_mul'],
                eta_mul=kwargs['eta_mul'],
                eta_min=kwargs['eta_min'],
                last_epoch=kwargs['last_epoch'])
            step_per_epoch = True

        self.scheduler = scheduler
        self.step_per_epoch = step_per_epoch
    
    def step(self, *args, **kwargs):
        self.scheduler.step(*args, **kwargs)

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict, strict=False):
        if strict:
            self.scheduler.load_state_dict(state_dict)
        else:
            try:
                self.scheduler.load_state_dict(state_dict)
            except:
                return
