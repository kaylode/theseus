import torch
import torchvision
from torchvision.transforms import functional as TFF
import matplotlib.pyplot as plt
from theseus.base.trainer.supervised_trainer import SupervisedTrainer
from theseus.utilities.loading import load_state_dict
from theseus.base.augmentations.custom import Denormalize
from theseus.utilities.visualization.visualizer import Visualizer

from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

class SegmentationTrainer(SupervisedTrainer):
    """Trainer for segmentation tasks
    
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def check_best(self, metric_dict):
        """
        Hook function, called after metrics are calculated
        """
        if metric_dict['dice'] > self.best_value:
            if self.iters > 0: # Have been training, else in evaluation-only mode or just sanity check
                LOGGER.text(
                    f"Evaluation improved from {self.best_value} to {metric_dict['dice']}",
                    level=LoggerObserver.INFO)
                self.best_value = metric_dict['dice']
                self.save_checkpoint('best')
            
            else:
                if self.visualize_when_val:
                    self.visualize_pred()

    def save_checkpoint(self, outname='last'):
        """
        Save all information of the current iteration
        """
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

    def load_checkpoint(self, path:str):
        """
        Load all information the current iteration from checkpoint 
        """
        LOGGER.text("Loading checkpoints...", level=LoggerObserver.INFO)
        state_dict = torch.load(path)
        self.epoch = load_state_dict(self.epoch, state_dict, 'epoch')
        self.start_iter = load_state_dict(self.start_iter, state_dict, 'iters')
        self.best_value = load_state_dict(self.best_value, state_dict, 'best_value')  
        self.scaler = load_state_dict(self.scaler, state_dict, self.scaler.state_dict_key)

        
    def visualize_gt(self):
        """
        Visualize dataloader for sanity check 
        """

        LOGGER.text("Visualizing dataset...", level=LoggerObserver.DEBUG)
        denom = Denormalize()
        batch = next(iter(self.trainloader))
        images = batch["inputs"]

        batch = []
        for idx, inputs in enumerate(images):
            img_show = denom(inputs)
            img_cam = TFF.to_tensor(img_show)
            batch.append(img_cam)
        batch = torch.stack(batch, dim=0)
        grid_img = torchvision.utils.make_grid(batch, nrow=int((idx+1)/8), normalize=False)

        fig = plt.figure(figsize=(8,8))
        plt.tight_layout(pad=0)
        plt.axis('off')
        plt.imshow(grid_img.permute(1, 2, 0))
        LOGGER.log([{
            'tag': "Sanitycheck/batch/train",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': self.iters
            }
        }])

        

        batch = next(iter(self.valloader))
        images = batch["inputs"]

        batch = []
        for idx, inputs in enumerate(images):
            img_show = denom(inputs)
            img_cam = TFF.to_tensor(img_show)
            batch.append(img_cam)
        batch = torch.stack(batch, dim=0)
        grid_img = torchvision.utils.make_grid(batch, nrow=int((idx+1)/8), normalize=False)

        fig = plt.figure(figsize=(8,8))
        plt.tight_layout(pad=0)
        plt.axis('off')
        plt.imshow(grid_img.permute(1, 2, 0))

        LOGGER.log([{
            'tag': "Sanitycheck/batch/val",
            'value': fig,
            'type': LoggerObserver.FIGURE,
            'kwargs': {
                'step': self.iters
            }
        }])

    def visualize_pred(self):
        r"""Visualize model prediction 
        
        """
        pass
        # Vizualize model predictions
        # LOGGER.text("Visualizing model predictions...", level=LoggerObserver.DEBUG)

        # visualizer = Visualizer()

        # denom = Denormalize()
        # batch = next(iter(self.valloader))
        # images = batch["inputs"]
        # targets = batch["targets"]

        # self.model.eval()

        # model_name = self.model.model.name
        

    @torch.no_grad()
    def visualize_model(self):
        # Vizualize Model Graph
        LOGGER.text("Visualizing architecture...", level=LoggerObserver.DEBUG)

        batch = next(iter(self.valloader))
        images = batch["inputs"].to(self.model.device)
        LOGGER.log([{
            'tag': "Sanitycheck/analysis/architecture",
            'value': self.model.model,
            'type': LoggerObserver.TORCH_MODULE,
            'kwargs': {
                'inputs': images
            }
        }])

    def analyze_gt(self):
        """
        Perform simple data analysis
        """
        pass

    def on_evaluate_end(self):
        if self.visualize_when_val:
            self.visualize_pred()
        self.save_checkpoint()
    
    def on_start(self):
        if self.resume is not None:
            self.load_checkpoint(self.resume)

    def sanitycheck(self):
        """Sanity check before training
        """
        self.visualize_gt()
        self.analyze_gt()
        self.visualize_model()
        self.evaluate_epoch()