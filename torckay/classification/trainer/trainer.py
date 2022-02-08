import torch
from torckay.base.trainer.supervised_trainer import SupervisedTrainer
from torckay.utilities.loggers.logger import LoggerManager
from torckay.utilities.loading import load_state_dict
LOGGER = LoggerManager.init_logger(__name__)

class ClassificationTrainer(SupervisedTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def check_best(self, metric_dict):
        if metric_dict['acc'] > self.best_value:
            self.save_checkpoint('best')

    def save_checkpoint(self, outname='last'):
        weights = {
            'model': self.model.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'iters': self.iters,
            'best_value': self.best_value,
        }

        if self.scaler is not None:
            weights[self.scaler.state_dict_key] = self.scaler.state_dict()
           
        self.checkpoint.save(weights, outname)

    def load_checkpoint(self, path):
        state_dict = torch.load(path)
        self.model = load_state_dict(self.model, state_dict, 'model')
        self.optimizer = load_state_dict(self.optimizer, state_dict, 'optimizer')
        self.scaler = load_state_dict(self.scaler, state_dict, self.scaler.state_dict_key)
        self.epoch = load_state_dict(self.epoch, state_dict, 'epoch')
        self.start_iter = load_state_dict(self.start_iter, state_dict, 'iters')
        self.best_value = load_state_dict(self.best_value, state_dict, 'best_value')
        
    def visualize_batch(self):
        raise NotImplementedError

    def on_evaluate_end(self):
        if self.visualize_when_val:
            self.visualize_batch()
        self.save_checkpoint()
    
    def on_training_start(self):
        if self.resume is not None:
            self.load_checkpoint(self.resume)
        

    def on_training_end(self):
        return

    def on_evaluate_end(self):
        return