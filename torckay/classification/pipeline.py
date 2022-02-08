from typing import Callable, Dict, Optional

import torch
from torckay.base.models.wrapper import ModelWithLoss
from torckay.opt import Opts
from torckay.base.optimizers import OPTIM_REGISTRY
from torckay.base.optimizers.schedulers import SCHEDULER_REGISTRY
from torckay.base.augmentations.torchvision import TRANSFORM_REGISTRY
from torckay.classification.losses import LOSS_REGISTRY
from torckay.classification.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from torckay.classification.trainer import TRAINER_REGISTRY
from torckay.classification.metrics import METRIC_REGISTRY
from torckay.classification.models import MODEL_REGISTRY
from torckay.utilities.getter import (get_instance, get_instance_recursively)
from torckay.utilities.loading import load_yaml
from torckay.base.optimizers.scalers import NativeScaler

from torckay.classification.datasets import CSVLoader

from torckay.utilities.loggers.logger import LoggerManager
 
LOGGER = LoggerManager.init_logger(__name__)

class Pipeline(object):
    """docstring for Pipeline."""

    def __init__(
        self,
        opt: Opts
    ):
        super(Pipeline, self).__init__()
        self.opt = opt
        
        self.transform_cfg = load_yaml(opt['global']['cfg_transform'])

        self.device = torch.device(opt['device'])

        self.transform = get_instance_recursively(
            self.transform_cfg, registry=TRANSFORM_REGISTRY
        )

        self.train_dataset = get_instance(
            opt['data']["dataset"]['train'],
            registry=DATASET_REGISTRY,
            transform=self.transform['train'],
        )

        self.val_dataset = get_instance(
            opt['data']["dataset"]['val'],
            registry=DATASET_REGISTRY,
            transform=self.transform['val'],
        )

        CLASSNAMES = self.val_dataset.classnames

        self.train_dataloader = get_instance(
            opt['data']["dataloader"]['train'],
            registry=DATALOADER_REGISTRY,
            dataset=self.train_dataset,
        )

        self.val_dataloader = get_instance(
            opt['data']["dataloader"]['val'],
            registry=DATALOADER_REGISTRY,
            dataset=self.val_dataset
        )

        model = get_instance(self.opt["model"], registry=MODEL_REGISTRY, classnames=CLASSNAMES).to(self.device)
        criterion = get_instance(self.opt["loss"], registry=LOSS_REGISTRY).to(
            self.device
        )
        self.model = ModelWithLoss(model, criterion)

        self.metrics = get_instance_recursively(self.opt['metrics'], registry=METRIC_REGISTRY)

        self.optimizer = get_instance(
            self.opt["optimizer"],
            registry=OPTIM_REGISTRY,
            params=self.model.parameters(),
        )

        self.scheduler = get_instance(
            self.opt["scheduler"], registry=SCHEDULER_REGISTRY, optimizer=self.optimizer
            **{
                'num_epochs': self.opt["trainer"]['args']['num_epochs'],
                'trainset': self.train_dataset,
                'batch_size': self.opt["data"]['dataloader']['val']['args']['batch_size'],
                'train_len': len(self.train_dataloader),
            }
        )

        self.scaler = NativeScaler()

        self.learner = get_instance(
            self.opt["trainer"],
            model=self.model,
            trainloader=self.train_dataloader,
            valloader=self.val_dataloader,
            metrics=self.metrics,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            registry=TRAINER_REGISTRY,
        )

    def infocheck(self):
        LOGGER.info(f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        LOGGER.info(self.opt)

    def sanitycheck(self):
        self.infocheck()
        LOGGER.info("Sanity checking before training")
        self.evaluate()

    def fit(self):
        self.sanitycheck()
        self.learner.fit()

    def evaluate(self):
        LOGGER.info("Evaluating ")
   

  