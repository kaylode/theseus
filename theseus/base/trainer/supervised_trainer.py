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

        self.step_per_epoch = self.scheduler.step_per_epoch


    def training_epoch(self):
        """
        Perform training one epoch
        """
        self.model.train()

        running_loss = {}
        running_time = 0

        self.optimizer.zero_grad()
        for i, batch in enumerate(self.trainloader):
            
            start_time = time.time()

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

            self.optimizer.zero_grad()

            torch.cuda.synchronize()
            end_time = time.time()

            for (key,value) in loss_dict.items():
                if key in running_loss.keys():
                    running_loss[key] += value
                else:
                    running_loss[key] = value

            running_time += end_time-start_time

            # Calculate current iteration
            self.iters = self.iters + 1

            # Logging
            if self.iters % self.print_interval == 0:
                for key in running_loss.keys():
                    running_loss[key] /= self.print_interval
                    running_loss[key] = np.round(running_loss[key], 5)
                loss_string = '{}'.format(running_loss)[1:-1].replace("'",'').replace(",",' ||')

                LOGGER.text(
                    "[{}|{}] || {} || Time: {:10.4f} (it/s)".format(
                        self.iters, self.num_iterations,
                        loss_string, self.print_interval/running_time), 
                    LoggerObserver.INFO)
                
                log_dict = [{
                    'tag': f"Training/{k} Loss",
                    'value': v/self.print_interval,
                    'type': LoggerObserver.SCALAR,
                    'kwargs': {
                        'step': self.iters
                    }
                } for k,v in running_loss.items()]


                log_dict.append({
                    'tag': f"Training/Iterations per second",
                    'value': self.print_interval/running_time,
                    'type': LoggerObserver.SCALAR,
                    'kwargs': {
                        'step': self.iters
                    }
                })
                LOGGER.log(log_dict)

                running_loss = {}
                running_time = 0

            # Saving checkpoint
            if (self.iters % self.save_interval == 0 or self.iters == self.num_iterations - 1):
                LOGGER.text(f'Save model at [{self.iters}|{self.num_iterations}] to last.pth', LoggerObserver.INFO)
                self.save_checkpoint()

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