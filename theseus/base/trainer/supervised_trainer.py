import time
import numpy as np
from tqdm import tqdm
import torch
from torch.cuda import amp

from .base_trainer import BaseTrainer

from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

class SupervisedTrainer(BaseTrainer):
    """Trainer for supervised tasks
    
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

    """
    def __init__(
        self, 
        model, 
        trainloader, 
        valloader,
        metrics,
        optimizer,
        scheduler,
        **kwargs):

        super().__init__(**kwargs)

        self.model = model
        self.metrics = metrics 
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainloader = trainloader
        self.valloader = valloader
        self.use_cuda = next(self.model.parameters()).is_cuda

        self.step_per_epoch = self.scheduler.step_per_epoch


    def training_epoch(self):
        """
        Perform training one epoch
        """
        self.model.train()
        self.callbacks.run('on_train_epoch_start')
        self.optimizer.zero_grad()
        for i, batch in enumerate(self.trainloader):
            self.callbacks.run('on_train_batch_start', {'batch': batch})

            # Gradient scaler
            with amp.autocast(enabled=self.use_amp):
                outputs = self.model.training_step(batch)
                loss = outputs['loss']
                loss_dict = outputs['loss_dict']

            # Backward loss
            self.scaler(loss, self.optimizer)
            
            # Optmizer step
            self.scaler.step(self.optimizer, clip_grad=self.clip_grad, parameters=self.model.parameters())
            if not self.step_per_epoch:
                self.scheduler.step()
            self.optimizer.zero_grad()

            if self.use_cuda:
                torch.cuda.synchronize()

            # Calculate current iteration
            self.iters = self.iters + 1

            # Get learning rate
            lrl = [x['lr'] for x in self.optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            self.callbacks.run('on_train_batch_end', {
                'loss_dict': loss_dict,
                'iters': self.iters,
                'num_iterations': self.num_iterations,
                'lr': lr
            })


        self.callbacks.run('on_train_epoch_end')
        

    @torch.no_grad()   
    def evaluate_epoch(self):
        """
        Perform validation one epoch
        """
        self.model.eval()
        epoch_loss = {}

        metric_dict = {}
        LOGGER.text('=============================EVALUATION===================================', LoggerObserver.INFO)

        start_time = time.time()

        # Gradient scaler
        with amp.autocast(enabled=self.use_amp):
            for batch in tqdm(self.valloader):
                outputs = self.model.evaluate_step(batch, self.metrics)
                
                loss_dict = outputs['loss_dict']
                for (key,value) in loss_dict.items():
                    if key in epoch_loss.keys():
                        epoch_loss[key] += value
                    else:
                        epoch_loss[key] = value

        end_time = time.time()
        running_time = end_time - start_time
             
        metric_dict = {}
        for metric in self.metrics:
            metric_dict.update(metric.value())
            metric.reset()  

        # Logging
        for key in epoch_loss.keys():
            epoch_loss[key] /= len(self.valloader)
            epoch_loss[key] = np.round(epoch_loss[key], 5)
        loss_string = '{}'.format(epoch_loss)[1:-1].replace("'",'').replace(",",' ||')
        LOGGER.text(
            "[{}|{}] || {} || Time: {:10.4f} (it/s)".format(
                self.iters, self.num_iterations, loss_string, len(self.valloader)/running_time),
        level=LoggerObserver.INFO)

        metric_string = ""
        for metric, score in metric_dict.items():
            if isinstance(score, (int, float)):
                metric_string += metric +': ' + f"{score:.5f}" +' | '
        metric_string +='\n'

        LOGGER.text(metric_string, level=LoggerObserver.INFO)
        LOGGER.text('==========================================================================', level=LoggerObserver.INFO)

        log_dict = [{
            'tag': f"Validation/{k} Loss",
            'value': v/len(self.valloader),
            'type': LoggerObserver.SCALAR,
            'kwargs': {
                'step': self.iters
            }
        } for k,v in epoch_loss.items()]

        log_dict += [{
            'tag': f"Validation/{k}",
            'value': v,
            'kwargs': {
                'step': self.iters
            }
        } for k,v in metric_dict.items()]

        LOGGER.log(log_dict)

        # Hook function
        self.check_best(metric_dict)

    def check_best(self, metric_dict):
        return 

    def on_start(self):
        # Init scheduler params
        if self.step_per_epoch:
            self.scheduler.last_epoch = self.iters//len(self.trainloader) - 1

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
                    'step': self.iters
                }
            }])

    def on_finish(self):
        self.save_checkpoint()