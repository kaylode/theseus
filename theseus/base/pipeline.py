from datetime import datetime

import os
import torch
from theseus.base.models.wrapper import ModelWithLoss
from theseus.opt import Config
from theseus.base.optimizers import OPTIM_REGISTRY, SCHEDULER_REGISTRY
from theseus.base.augmentations import TRANSFORM_REGISTRY
from theseus.base.losses import LOSS_REGISTRY
from theseus.base.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from theseus.base.trainer import TRAINER_REGISTRY
from theseus.base.metrics import METRIC_REGISTRY
from theseus.base.models import MODEL_REGISTRY
from theseus.base.callbacks import CALLBACKS_REGISTRY
from theseus.utilities.getter import (get_instance, get_instance_recursively)
from theseus.utilities.loggers import LoggerObserver, FileLogger, ImageWriter
from theseus.utilities.loading import load_state_dict
from theseus.utilities.folder import get_new_folder_name
from theseus.utilities.cuda import get_devices_info, move_to, get_device

class BasePipeline(object):
    """docstring for BasePipeline."""

    def __init__(
        self,
        opt: Config
    ):
        super(BasePipeline, self).__init__()
        self.opt = opt

    def init_globals(self):
        # Main Loggers
        self.logger = LoggerObserver.getLogger("main") 

        # Global variables
        self.exp_name = self.opt['global']['exp_name']
        self.exist_ok = self.opt['global']['exist_ok']
        self.debug = self.opt['global']['debug']
        self.device_name = self.opt['global']['device']
        self.transform_cfg = Config.load_yaml(self.opt['global']['cfg_transform'])
        self.device = get_device(self.device_name)
        
        # Experiment name
        if self.exp_name:
            self.savedir = os.path.join(self.opt['global']['save_dir'], self.exp_name)
            if not self.exist_ok:
                self.savedir = get_new_folder_name(self.savedir)
        else:
            self.savedir = os.path.join(self.opt['global']['save_dir'], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.savedir, exist_ok=True)

        # Logging to files
        file_logger = FileLogger(__name__, self.savedir, debug=self.debug)
        self.logger.subscribe(file_logger)
        self.logger.text(self.opt, level=LoggerObserver.INFO)
        self.logger.text(f"Everything will be saved to {self.savedir}", level=LoggerObserver.INFO)
    
    def init_train_dataloader(self):
        # DataLoaders
        self.transform = get_instance_recursively(
            self.transform_cfg, registry=self.transform_registry
        )
        self.train_dataset = get_instance_recursively(
            self.opt['data']["dataset"]['train'],
            registry=self.dataset_registry,
            transform=self.transform['train'],
        )
        self.train_dataloader = get_instance_recursively(
            self.opt['data']["dataloader"]['train'],
            registry=self.dataloader_registry,
            dataset=self.train_dataset,
        )

        self.logger.text(f"Number of training samples: {len(self.train_dataset)}", level=LoggerObserver.INFO)
        self.logger.text(f"Number of training iterations each epoch: {len(self.train_dataloader)}", level=LoggerObserver.INFO)


    def init_validation_dataloader(self):
        # Transforms & Datasets
        self.transform = get_instance_recursively(
            self.transform_cfg, registry=self.transform_registry
        )
        self.val_dataset = get_instance_recursively(
            self.opt['data']["dataset"]['val'],
            registry=self.dataset_registry,
            transform=self.transform['val'],
        )
        self.val_dataloader = get_instance_recursively(
            self.opt['data']["dataloader"]['val'],
            registry=self.dataloader_registry,
            dataset=self.val_dataset
        )

        self.logger.text(f"Number of validation samples: {len(self.val_dataset)}", level=LoggerObserver.INFO)
        self.logger.text(f"Number of validation iterations each epoch: {len(self.val_dataloader)}", level=LoggerObserver.INFO)

    def init_model(self):
        CLASSNAMES = self.val_dataset.classnames
        model = get_instance(
            self.opt["model"], 
            registry=self.model_registry, 
            num_classes=len(CLASSNAMES),
            classnames=CLASSNAMES)
        model = move_to(model, self.device)
        return model

    def init_criterion(self):
        criterion = get_instance_recursively(
            self.opt["loss"], 
            registry=self.loss_registry)
        criterion = move_to(criterion, self.device)
        return criterion

    def init_model_with_loss(self):
        model = self.init_model()
        criterion = self.init_criterion()
        self.model = ModelWithLoss(model, criterion, self.device)
        self.logger.text(f"Number of trainable parameters: {self.model.trainable_parameters():,}", level=LoggerObserver.INFO)
        device_info = get_devices_info(self.device_name)
        self.logger.text("Using " + device_info, level=LoggerObserver.INFO)

    def init_metrics(self):
        CLASSNAMES = self.val_dataset.classnames
        self.metrics = get_instance_recursively(
            self.opt['metrics'], 
            registry=self.metric_registry, 
            classnames=CLASSNAMES)

    def init_optimizer(self):
        self.optimizer = get_instance(
            self.opt["optimizer"],
            registry=self.optimizer_registry,
            params=self.model.parameters(),
        )

    def init_loading(self):
        self.resume = self.opt['global']['resume']
        self.pretrained = self.opt['global']['pretrained']
        self.last_epoch = -1
        if self.pretrained:
            state_dict = torch.load(self.pretrained)
            self.model.model = load_state_dict(self.model.model, state_dict, 'model')

        if self.resume:
            state_dict = torch.load(self.resume)
            self.model.model = load_state_dict(self.model.model, state_dict, 'model')
            self.optimizer = load_state_dict(self.optimizer, state_dict, 'optimizer')
            iters = load_state_dict(None, state_dict, 'iters')
            self.last_epoch = iters//len(self.train_dataloader) - 1

    def init_scheduler(self):
        if "scheduler" in self.opt.keys():
            self.scheduler = get_instance(
                self.opt["scheduler"], registry=self.scheduler_registry, optimizer=self.optimizer,
                **{
                    'num_epochs': self.opt["trainer"]['args']['num_iterations'] // len(self.train_dataloader),
                    'trainset': self.train_dataset,
                    'batch_size': self.opt["data"]['dataloader']['val']['args']['batch_size'],
                    'last_epoch': self.last_epoch,
                }
            )

            if self.resume:
                state_dict = torch.load(self.resume)
                self.scheduler = load_state_dict(self.scheduler, state_dict, 'scheduler')
        else:
            self.scheduler = None

    def init_callbacks(self):
        callbacks = get_instance_recursively(
            self.opt["callbacks"],
            print_interval=self.opt["trainer"]['args']['print_interval'],
            save_interval=self.opt["trainer"]['args']['save_interval'],
            save_dir=self.savedir,
            resume=self.resume,
            config_dict=self.opt,
            registry=self.callbacks_registry
        )
        return callbacks

    def init_trainer(self, callbacks):
        self.trainer = get_instance(
            self.opt["trainer"],
            model=self.model,
            trainloader=getattr(self, "train_dataloader", None),
            valloader=getattr(self, "val_dataloader", None),
            metrics=getattr(self, "metrics", None),
            optimizer=getattr(self, "optimizer", None),
            scheduler=getattr(self, "scheduler", None),
            debug=self.debug,
            registry=self.trainer_registry,
            callbacks=callbacks
        )

    def save_configs(self):
        self.opt.save_yaml(os.path.join(self.savedir, 'pipeline.yaml'))
        self.transform_cfg.save_yaml(os.path.join(self.savedir, 'transform.yaml'))

    def init_registry(self):
        self.model_registry = MODEL_REGISTRY
        self.dataset_registry = DATASET_REGISTRY
        self.dataloader_registry = DATALOADER_REGISTRY
        self.metric_registry = METRIC_REGISTRY
        self.loss_registry = LOSS_REGISTRY
        self.optimizer_registry = OPTIM_REGISTRY
        self.scheduler_registry = SCHEDULER_REGISTRY
        self.callbacks_registry = CALLBACKS_REGISTRY
        self.trainer_registry = TRAINER_REGISTRY
        self.transform_registry = TRANSFORM_REGISTRY
        self.logger.text(
            "You should override the init_registry() function", LoggerObserver.CRITICAL
        )

    def init_pipeline(self, train=False):
        self.init_globals()
        self.init_registry()
        if train:
            self.init_train_dataloader()
            self.init_validation_dataloader()
            self.init_model_with_loss()
            self.init_metrics()
            self.init_optimizer()
            self.init_loading()
            self.init_scheduler()
            callbacks = self.init_callbacks()
            self.save_configs()
        else:
            self.init_validation_dataloader()
            self.init_model_with_loss()
            self.init_metrics()
            self.init_loading()
            callbacks = [self.callbacks_registry.get("LoggerCallbacks")()]
        self.init_trainer(callbacks)

    def fit(self):
        self.init_pipeline(train=True)
        self.trainer.fit()

    def evaluate(self):
        self.init_pipeline(train=False)
        writer = ImageWriter(os.path.join(self.savedir, 'samples'))
        self.logger.subscribe(writer)

        self.logger.text("Evaluating...", level=LoggerObserver.INFO)
        self.trainer.evaluate_epoch()

class BaseTestPipeline(object):
    def __init__(
            self,
            opt: Config
        ):

        super(BaseTestPipeline, self).__init__()
        self.opt = opt

    def init_globals(self):
        # Main Loggers
        self.logger = LoggerObserver.getLogger("main") 

        # Global variables
        self.exp_name = self.opt['global']['exp_name']
        self.exist_ok = self.opt['global']['exist_ok']
        self.debug = self.opt['global']['debug']
        self.device_name = self.opt['global']['device']
        self.transform_cfg = Config.load_yaml(self.opt['global']['cfg_transform'])
        self.device = get_device(self.device_name)
        
        # Experiment name
        if self.exp_name:
            self.savedir = os.path.join(self.opt['global']['save_dir'], self.exp_name, 'test')
            if not self.exist_ok:
                self.savedir = get_new_folder_name(self.savedir)
        else:
            self.savedir = os.path.join(self.opt['global']['save_dir'], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'test')
        os.makedirs(self.savedir, exist_ok=True)

        # Logging to files
        file_logger = FileLogger(__name__, self.savedir, debug=self.debug)
        self.logger.subscribe(file_logger)
        self.logger.text(self.opt, level=LoggerObserver.INFO)
        self.logger.text(f"Everything will be saved to {self.savedir}", level=LoggerObserver.INFO)

    def init_registry(self):
        self.model_registry = MODEL_REGISTRY
        self.dataset_registry = DATASET_REGISTRY
        self.dataloader_registry = DATALOADER_REGISTRY
        self.transform_registry = TRANSFORM_REGISTRY
        self.logger.text(
            "You should override the init_registry() function", LoggerObserver.INFO
        )

    def init_test_dataloader(self):
        # Transforms & Datasets
        self.transform = get_instance_recursively(
            self.transform_cfg, registry=self.transform_registry
        )

        self.dataset = get_instance(
            self.opt['data']["dataset"],
            registry=DATASET_REGISTRY,
            transform=self.transform['test'],
        )
        
        self.dataloader = get_instance(
            self.opt['data']["dataloader"],
            registry=DATALOADER_REGISTRY,
            dataset=self.dataset,
        )

        self.logger.text(f"Number of test samples: {len(self.dataset)}", level=LoggerObserver.INFO)
        self.logger.text(f"Number of test iterations each epoch: {len(self.dataloader)}", level=LoggerObserver.INFO)

    def init_loading(self):
        self.weights = self.opt['global']['weights']
        if self.weights:
            state_dict = torch.load(self.weights)
            self.model = load_state_dict(self.model, state_dict, 'model')

    def init_model(self):
        CLASSNAMES = self.dataset.classnames
        self.model = get_instance(
            self.opt["model"], 
            registry=MODEL_REGISTRY, 
            num_classes = len(CLASSNAMES),
            classnames=CLASSNAMES)
        self.model = move_to(self.model, self.device)
        self.model.eval()

    def init_pipeline(self):
        self.init_globals()
        self.init_registry()
        self.init_test_dataloader()
        self.init_model()
        self.init_loading()
     
    def inference(self):
        raise NotImplementedError()
        