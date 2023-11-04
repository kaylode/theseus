import os
from datetime import datetime

import torch
from omegaconf import DictConfig, OmegaConf

from theseus.base.augmentations import TRANSFORM_REGISTRY
from theseus.base.callbacks import CALLBACKS_REGISTRY
from theseus.base.datasets import (
    DATALOADER_REGISTRY,
    DATASET_REGISTRY,
    LightningDataModuleWrapper,
)
from theseus.base.losses import LOSS_REGISTRY
from theseus.base.metrics import METRIC_REGISTRY
from theseus.base.models import MODEL_REGISTRY, LightningModelWrapper
from theseus.base.trainer import TRAINER_REGISTRY
from theseus.base.utilities.folder import get_new_folder_name
from theseus.base.utilities.getter import get_instance, get_instance_recursively
from theseus.base.utilities.loggers import FileLogger, ImageWriter, LoggerObserver
from theseus.base.utilities.seed import seed_everything


class BasePipeline(object):
    """docstring for BasePipeline."""

    def __init__(self, opt: DictConfig):
        super(BasePipeline, self).__init__()
        self.opt = opt
        self.seed = self.opt["global"].get("seed", 1702)
        seed_everything(self.seed)

        self.initialized = False

    def init_globals(self):
        # Main Loggers
        self.logger = LoggerObserver.getLogger("main")

        # Global variables
        self.exp_name = self.opt["global"].get("exp_name", None)
        self.exist_ok = self.opt["global"].get("exist_ok", False)
        self.debug = self.opt["global"].get("debug", False)
        self.resume = self.opt["global"].get("resume", None)
        self.pretrained = self.opt["global"].get("pretrained", None)
        self.transform_cfg = self.opt["global"].get("cfg_transform", None)

        # Experiment name
        if self.exp_name:
            self.savedir = os.path.join(
                self.opt["global"].get("save_dir", "runs"), self.exp_name
            )
            if not self.exist_ok:
                self.savedir = get_new_folder_name(self.savedir)
        else:
            self.savedir = os.path.join(
                self.opt["global"].get("save_dir", "runs"),
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            )
        os.makedirs(self.savedir, exist_ok=True)

        # Logging to files
        file_logger = FileLogger(__name__, self.savedir, debug=self.debug)
        self.logger.subscribe(file_logger)

        # Logging images
        image_logger = ImageWriter(self.savedir)
        self.logger.subscribe(image_logger)

        self.transform_cfg = self.opt.get("augmentations", None)

        # Logging out configs
        self.logger.text("\n" + OmegaConf.to_yaml(self.opt), level=LoggerObserver.INFO)
        self.logger.text(
            f"Everything will be saved to {self.savedir}",
            level=LoggerObserver.INFO,
        )

    def init_train_dataloader(self):
        # DataLoaders
        if self.transform_cfg is not None:
            self.transform = get_instance_recursively(
                self.transform_cfg, registry=self.transform_registry
            )
        else:
            self.transform = {"train": None, "val": None}

        self.train_dataset = get_instance_recursively(
            self.opt["data"]["dataset"]["train"],
            registry=self.dataset_registry,
            transform=self.transform["train"],
        )
        self.train_dataloader = get_instance_recursively(
            self.opt["data"]["dataloader"]["train"],
            registry=self.dataloader_registry,
            dataset=self.train_dataset,
        )

        self.logger.text(
            f"Number of training samples: {len(self.train_dataset)}",
            level=LoggerObserver.INFO,
        )
        self.logger.text(
            f"Number of training iterations each epoch: {len(self.train_dataloader)}",
            level=LoggerObserver.INFO,
        )

    def init_validation_dataloader(self):
        # DataLoaders
        if self.transform_cfg is not None:
            self.transform = get_instance_recursively(
                self.transform_cfg, registry=self.transform_registry
            )
        else:
            self.transform = {"train": None, "val": None}

        self.val_dataset = get_instance_recursively(
            self.opt["data"]["dataset"]["val"],
            registry=self.dataset_registry,
            transform=self.transform["val"],
        )
        self.val_dataloader = get_instance_recursively(
            self.opt["data"]["dataloader"]["val"],
            registry=self.dataloader_registry,
            dataset=self.val_dataset,
        )

        self.logger.text(
            f"Number of validation samples: {len(self.val_dataset)}",
            level=LoggerObserver.INFO,
        )
        self.logger.text(
            f"Number of validation iterations each epoch: {len(self.val_dataloader)}",
            level=LoggerObserver.INFO,
        )

    def init_datamodule(self):
        self.datamodule = LightningDataModuleWrapper(
            trainloader=getattr(self, "train_dataloader", None),
            valloader=getattr(self, "val_dataloader", None),
            testloader=getattr(self, "test_dataloader", None),
        )

    def init_model(self):
        CLASSNAMES = getattr(self.val_dataset, "classnames", None)
        model = get_instance(
            self.opt["model"],
            registry=self.model_registry,
            num_classes=len(CLASSNAMES) if CLASSNAMES is not None else None,
            classnames=CLASSNAMES,
        )
        return model

    def init_criterion(self):
        CLASSNAMES = getattr(self.val_dataset, "classnames", None)
        self.criterion = get_instance_recursively(
            self.opt["loss"],
            registry=self.loss_registry,
            num_classes=len(CLASSNAMES) if CLASSNAMES is not None else None,
            classnames=CLASSNAMES,
        )
        return self.criterion

    def init_model_with_loss(self, is_train=True):
        self.model = self.init_model()
        criterion = self.init_criterion()
        num_epochs = self.opt["trainer"]["args"]["max_epochs"]
        batch_size = self.opt["data"]["dataloader"]["val"]["args"]["batch_size"]

        self.model = LightningModelWrapper(
            self.model,
            criterion,
            datamodule=getattr(self, "datamodule", None),
            metrics=getattr(self, "metrics", None),
            optimizer_config=self.opt["optimizer"] if is_train else None,
            scheduler_config=self.opt["scheduler"] if is_train else None,
            scheduler_kwargs={
                "num_epochs": num_epochs,
                "num_iterations": num_epochs * len(self.train_dataloader),
                "batch_size": batch_size,
                "last_epoch": getattr(self, "last_epoch", -1),
            }
            if is_train
            else None,
        )

        pretrained = self.opt["global"].get("pretrained", None)
        if pretrained:
            state_dict = torch.load(pretrained, map_location="cpu")
            try:
                self.model.load_state_dict(state_dict["state_dict"], strict=False)
                self.logger.text(
                    f"Loaded pretrained model from {pretrained}",
                    level=LoggerObserver.SUCCESS,
                )
            except Exception as e:
                self.logger.text(
                    f"Loaded pretrained model from {pretrained}. Mismatched keys: {e}",
                    level=LoggerObserver.WARN,
                )

    def init_metrics(self):
        CLASSNAMES = getattr(self.val_dataset, "classnames", None)
        self.metrics = get_instance_recursively(
            self.opt["metrics"],
            registry=self.metric_registry,
            num_classes=len(CLASSNAMES) if CLASSNAMES is not None else None,
            classnames=CLASSNAMES,
        )

    def init_callbacks(self):
        callbacks = get_instance_recursively(
            self.opt["callbacks"],
            save_dir=getattr(self, "savedir", "runs"),
            resume=getattr(self, "resume", None),
            config_dict=self.opt,
            registry=self.callbacks_registry,
        )
        return callbacks

    def init_trainer(self, callbacks):
        self.trainer = get_instance(
            self.opt["trainer"],
            default_root_dir=getattr(self, "savedir", "runs"),
            deterministic="warn",
            callbacks=callbacks,
            registry=self.trainer_registry,
        )

    def save_configs(self):
        with open(os.path.join(self.savedir, "pipeline.yaml"), "w") as f:
            OmegaConf.save(config=self.opt, f=f)

    def init_registry(self):
        self.model_registry = MODEL_REGISTRY
        self.dataset_registry = DATASET_REGISTRY
        self.dataloader_registry = DATALOADER_REGISTRY
        self.metric_registry = METRIC_REGISTRY
        self.loss_registry = LOSS_REGISTRY
        self.callbacks_registry = CALLBACKS_REGISTRY
        self.trainer_registry = TRAINER_REGISTRY
        self.transform_registry = TRANSFORM_REGISTRY
        self.logger.text(
            "You should override the init_registry() function",
            LoggerObserver.CRITICAL,
        )

    def init_pipeline(self, train=False):
        if self.initialized:
            return
        self.init_globals()
        self.init_registry()
        if train:
            self.init_train_dataloader()
            self.init_validation_dataloader()
            self.init_datamodule()
            self.init_metrics()
            self.init_model_with_loss()
            callbacks = self.init_callbacks()
            self.save_configs()
        else:
            self.init_validation_dataloader()
            self.init_datamodule()
            self.init_metrics()
            self.init_model_with_loss(is_train=train)
            callbacks = []

        if getattr(self.model, "metrics", None):
            callbacks.insert(
                0,
                self.callbacks_registry.get("MetricLoggerCallback")(
                    save_dir=self.savedir
                ),
            )
        if getattr(self.model, "criterion", None):
            callbacks.insert(
                0,
                self.callbacks_registry.get("LossLoggerCallback")(
                    print_interval=self.opt["trainer"]["args"].get(
                        "log_every_n_steps", None
                    ),
                ),
            )
        callbacks.insert(0, self.callbacks_registry.get("TimerCallback")())

        self.init_trainer(callbacks)
        self.initialized = True

    def fit(self):
        self.init_pipeline(train=True)
        self.trainer.fit(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=self.resume,
        )

    def evaluate(self):
        self.init_pipeline(train=False)
        self.trainer.validate(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=self.resume,
        )

        return self.trainer.callback_metrics


class BaseTestPipeline(object):
    def __init__(self, opt: DictConfig):

        super(BaseTestPipeline, self).__init__()
        self.opt = opt
        self.seed = self.opt["global"].get("seed", 1702)
        seed_everything(self.seed)

    def init_globals(self):
        # Main Loggers
        self.logger = LoggerObserver.getLogger("main")

        # Global variables
        self.exp_name = self.opt["global"].get("exp_name", None)
        self.exist_ok = self.opt["global"].get("exist_ok", False)
        self.debug = self.opt["global"].get("debug", False)
        self.transform_cfg = self.opt["global"].get("cfg_transform", None)

        # Experiment name
        if self.exp_name:
            self.savedir = os.path.join(
                self.opt["global"].get("save_dir", "runs"), self.exp_name
            )
            if not self.exist_ok:
                self.savedir = get_new_folder_name(self.savedir)
        else:
            self.savedir = os.path.join(
                self.opt["global"].get("save_dir", "runs"),
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            )
        os.makedirs(self.savedir, exist_ok=True)

        self.transform_cfg = self.opt.get("augmentations", None)

        # Logging to files
        file_logger = FileLogger(__name__, self.savedir, debug=self.debug)
        self.logger.subscribe(file_logger)
        self.logger.text(self.opt, level=LoggerObserver.INFO)
        self.logger.text(
            f"Everything will be saved to {self.savedir}",
            level=LoggerObserver.INFO,
        )

    def init_registry(self):
        self.model_registry = MODEL_REGISTRY
        self.dataset_registry = DATASET_REGISTRY
        self.dataloader_registry = DATALOADER_REGISTRY
        self.transform_registry = TRANSFORM_REGISTRY
        self.logger.text(
            "You should override the init_registry() function",
            LoggerObserver.INFO,
        )

    def init_test_dataloader(self):
        # Transforms & Datasets
        self.transform = get_instance_recursively(
            self.transform_cfg, registry=self.transform_registry
        )

        transform_cfg = (
            self.transform["test"]
            if "test" in self.transform
            else self.transform["val"]
        )

        self.dataset = get_instance(
            self.opt["data"]["dataset"],
            registry=DATASET_REGISTRY,
            transform=transform_cfg,
        )

        self.dataloader = get_instance(
            self.opt["data"]["dataloader"],
            registry=DATALOADER_REGISTRY,
            dataset=self.dataset,
        )

        self.logger.text(
            f"Number of test samples: {len(self.dataset)}",
            level=LoggerObserver.INFO,
        )
        self.logger.text(
            f"Number of test iterations each epoch: {len(self.dataloader)}",
            level=LoggerObserver.INFO,
        )

    def init_model(self):
        CLASSNAMES = getattr(self.dataset, "classnames", None)
        self.model = get_instance(
            self.opt["model"],
            registry=MODEL_REGISTRY,
            num_classes=len(CLASSNAMES) if CLASSNAMES is not None else None,
            classnames=CLASSNAMES,
        )
        self.model = LightningModelWrapper(self.model)
        self.model.eval()

    def init_loading(self):
        self.weights = self.opt["global"].get("pretrained", None)
        if self.weights:
            state_dict = torch.load(self.weights, map_location="cpu")
            self.model.load_state_dict(state_dict["state_dict"])

    def init_pipeline(self):
        self.init_globals()
        self.init_registry()
        self.init_test_dataloader()
        self.init_model()
        self.init_loading()

    def inference(self):
        raise NotImplementedError()
