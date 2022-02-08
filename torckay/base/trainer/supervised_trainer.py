import time
import numpy as np
from tqdm import tqdm

import torch
from torch.cuda import amp

from .base_trainer import BaseTrainer
from torckay.utilities.loggers.logger import LoggerManager

LOGGER = LoggerManager.init_logger(__name__)

class SupervisedTrainer(BaseTrainer):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def training_epoch(self):
        self.model.train()

        running_loss = {}
        running_time = 0

        self.optimizer.zero_grad()
        for i, batch in enumerate(self.trainloader):
            
            start_time = time.time()
            with amp.autocast(enabled=self.use_amp):
                outputs = self.model.training_step(batch)

                loss = outputs['loss']
                loss_dict = outputs['loss_dict']
                loss /= self.accumulate_steps

            self.scaler(loss, self.optimizer)
            
            if i % self.accumulate_steps == 0 or i == len(self.trainloader)-1:
                self.scaler.step(self.optimizer, clip_grad=self.clip_grad, parameters=self.model.parameters())
                self.optimizer.zero_grad()

                if not self.step_per_epoch:
                    self.scheduler.step()
                    lrl = [x['lr'] for x in self.optimizer.param_groups]
                    lr = sum(lrl) / len(lrl)
                    log_dict = {'Training/Learning rate': lr}
                    self.tf_logger.write_dict(log_dict, step=self.iters)

            torch.cuda.synchronize()
            end_time = time.time()

            for (key,value) in loss_dict.items():
                if key in running_loss.keys():
                    running_loss[key] += value
                else:
                    running_loss[key] = value

            running_time += end_time-start_time
            self.iters = self.start_iter + len(self.trainloader)*self.epoch + i + 1
            if self.iters % self.print_per_iter == 0:
                
                for key in running_loss.keys():
                    running_loss[key] /= self.print_per_iter
                    running_loss[key] = np.round(running_loss[key], 5)
                loss_string = '{}'.format(running_loss)[1:-1].replace("'",'').replace(",",' ||')
                LOGGER.info("[{}|{}] [{}|{}] || {} || Time: {:10.4f}s".format(self.epoch, self.num_epochs, self.iters, self.num_iters,loss_string, running_time))
                
                log_dict = {f"Training/{k} Loss": v/self.print_per_iter for k,v in running_loss.items()}
                self.tf_logger.write_dict(log_dict, step=self.iters)
                running_loss = {}
                running_time = 0

            if (self.iters % self.save_per_iter == 0 or self.iters == self.num_iters - 1):
                LOGGER.info(f'Save model at [{self.iters}|{self.num_iters}] to last.pth')
                self.save_checkpoint()
            
    def evaluate_epoch(self):
        self.model.eval()
        epoch_loss = {}

        metric_dict = {}
        LOGGER.info('=============================EVALUATION===================================')
        start_time = time.time()
        with torch.no_grad():
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

        for key in epoch_loss.keys():
            epoch_loss[key] /= len(self.valloader)
            epoch_loss[key] = np.round(epoch_loss[key], 5)
        loss_string = '{}'.format(epoch_loss)[1:-1].replace("'",'').replace(",",' ||')
        LOGGER.info("[{}|{}] || {} || Time: {:10.4f} s".format(self.epoch, self.num_epochs, loss_string, running_time))

        metric_string = ""
        for metric, score in metric_dict.items():
            metric_string += metric +': ' + f"{score}:.5f" +' | '
        metric_string +='\n'
        LOGGER.info(metric_string)
        LOGGER.info('==========================================================================')

        log_dict = {f"Validation/{k} Loss": v/len(self.valloader) for k,v in epoch_loss.items()}

        metric_log_dict = {f"Validation/{k}":v for k,v in metric_dict.items()}
        log_dict.update(metric_log_dict)
        self.tf_logger.write_dict(log_dict, step=self.epoch)

        self.check_best(metric_dict)

    def check_best(self, metric_dict):
        return 