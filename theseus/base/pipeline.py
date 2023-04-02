import os
from datetime import datetime

import torch

from theseus.base.augmentations import TRANSFORM_REGISTRY
from theseus.base.callbacks import CALLBACKS_REGISTRY
from theseus.base.datasets import DATALOADER_REGISTRY, DATASET_REGISTRY
from theseus.base.losses import LOSS_REGISTRY
from theseus.base.metrics import METRIC_REGISTRY
from theseus.base.models import MODEL_REGISTRY
from theseus.base.models.wrapper import ModelWithLoss
from theseus.base.optimizers import OPTIM_REGISTRY, SCHEDULER_REGISTRY
from theseus.base.trainer import TRAINER_REGISTRY
from theseus.base.utilities.cuda import get_device, get_devices_info, move_to
from theseus.base.utilities.folder import get_new_folder_name
from theseus.base.utilities.getter import get_instance, get_instance_recursively
from theseus.base.utilities.loading import load_state_dict
from theseus.base.utilities.loggers import FileLogger, ImageWriter, LoggerObserver
from theseus.base.utilities.seed import seed_everything
from theseus.opt import Config


class BasePipeline(object):
    """docstring for BasePipeline."""

    def __init__(self, opt: Config):
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
        self.device_name = self.opt["global"].get("device", "cpu")
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

        if self.transform_cfg is not None:
            self.logger.text(
                "cfg_transform is deprecated, please use 'includes' instead",
                level=LoggerObserver.WARN,
            )
            self.transform_cfg = Config.load_yaml(self.transform_cfg)
            self.opt["augmentations"] = self.transform_cfg
        else:
            self.transform_cfg = self.opt.get("augmentations", None)

        self.device = get_device(self.device_name)

        # Logging out configs
        self.logger.text(self.opt, level=LoggerObserver.INFO)
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

    def init_model(self):
        CLASSNAMES = getattr(self.val_dataset, "classnames", None)
        model = get_instance(
            self.opt["model"],
            registry=self.model_registry,
            num_classes=len(CLASSNAMES) if CLASSNAMES is not None else None,
            classnames=CLASSNAMES,
        )
        model = move_to(model, self.device)
        return model

    def init_criterion(self):
        CLASSNAMES = getattr(self.val_dataset, "classnames", None)
        self.criterion = get_instance_recursively(
            self.opt["loss"],
            registry=self.loss_registry,
            num_classes=len(CLASSNAMES) if CLASSNAMES is not None else None,
            classnames=CLASSNAMES,
        )
        self.criterion = move_to(self.criterion, self.device)
        return self.criterion

    def init_model_with_loss(self):
        model = self.init_model()
        criterion = self.init_criterion()
        self.model = ModelWithLoss(model, criterion, self.device)
        self.logger.text(
            f"Number of trainable parameters: {self.model.trainable_parameters():,}",
            level=LoggerObserver.INFO,
        )
        device_info = get_devices_info(self.device_name)
        self.logger.text("Using " + device_info, level=LoggerObserver.INFO)

    def init_metrics(self):
        CLASSNAMES = getattr(self.val_dataset, "classnames", None)
        self.metrics = get_instance_recursively(
            self.opt["metrics"],
            registry=self.metric_registry,
            num_classes=len(CLASSNAMES) if CLASSNAMES is not None else None,
            classnames=CLASSNAMES,
        )

    def init_optimizer(self):
        self.optimizer = get_instance(
            self.opt["optimizer"],
            registry=self.optimizer_registry,
            params=self.model.parameters(),
        )

    def init_loading(self):
        self.last_epoch = -1
        if getattr(self, "pretrained", None):
            state_dict = torch.load(self.pretrained, map_location="cpu")
            self.model.model = load_state_dict(self.model.model, state_dict, "model")

        if getattr(self, "resume", None):
            state_dict = torch.load(self.resume, map_location="cpu")
            self.model.model = load_state_dict(self.model.model, state_dict, "model")
            self.optimizer = load_state_dict(self.optimizer, state_dict, "optimizer")
            iters = load_state_dict(None, state_dict, "iters")
            self.last_epoch = iters // len(self.train_dataloader) - 1

    def init_scheduler(self):
        if "scheduler" in self.opt.keys() and self.opt["scheduler"] is not None:
            self.scheduler = get_instance(
                self.opt["scheduler"],
                registry=self.scheduler_registry,
                optimizer=self.optimizer,
                **{
                    "num_epochs": self.opt["trainer"]["args"]["num_iterations"]
                    // len(self.train_dataloader),
                    "trainset": self.train_dataset,
                    "batch_size": self.opt["data"]["dataloader"]["val"]["args"][
                        "batch_size"
                    ],
                    "last_epoch": getattr(self, "last_epoch", -1),
                },
            )

            if getattr(self, "resume", None):
                state_dict = torch.load(self.resume)
                self.scheduler = load_state_dict(
                    self.scheduler, state_dict, "scheduler"
                )
        else:
            self.scheduler = None

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
            model=self.model,
            trainloader=getattr(self, "train_dataloader", None),
            valloader=getattr(self, "val_dataloader", None),
            metrics=getattr(self, "metrics", None),
            optimizer=getattr(self, "optimizer", None),
            scheduler=getattr(self, "scheduler", None),
            debug=getattr(self, "debug", False),
            registry=self.trainer_registry,
            callbacks=callbacks,
        )

    def save_configs(self):
        self.opt.save_yaml(os.path.join(self.savedir, "pipeline.yaml"))

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
            callbacks = []

        if getattr(self, "metrics", None):
            callbacks.insert(
                0,
                self.callbacks_registry.get("MetricLoggerCallbacks")(
                    save_dir=self.savedir
                ),
            )
        if getattr(self, "criterion", None):
            callbacks.insert(
                0,
                self.callbacks_registry.get("LossLoggerCallbacks")(
                    print_interval=self.opt["global"].get("print_interval", None),
                ),
            )
        if self.debug:
            callbacks.insert(0, self.callbacks_registry.get("DebugCallbacks")())
        callbacks.insert(0, self.callbacks_registry.get("TimerCallbacks")())
        self.init_trainer(callbacks)
        self.initialized = True

    def fit(self):
        self.init_pipeline(train=True)
        self.trainer.fit()

    def evaluate(self):
        self.init_pipeline(train=False)
        self.logger.text("Evaluating...", level=LoggerObserver.INFO)
        return self.trainer.evaluate_epoch()


class BaseTestPipeline(object):
    def __init__(self, opt: Config):

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
        self.device_name = self.opt["global"].get("device", "cpu")
        self.transform_cfg = self.opt["global"].get("cfg_transform", None)
        self.device = get_device(self.device_name)

        # Experiment name
        if self.exp_name:
            self.savedir = os.path.join(
                self.opt["global"].get("save_dir", "tests"), self.exp_name
            )
            if not self.exist_ok:
                self.savedir = get_new_folder_name(self.savedir)
        else:
            self.savedir = os.path.join(
                self.opt["global"].get("save_dir", "tests"),
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            )
        os.makedirs(self.savedir, exist_ok=True)

        if self.transform_cfg is not None:
            self.logger.text(
                "cfg_transform is deprecated, please use 'includes' instead",
                level=LoggerObserver.WARN,
            )
            self.transform_cfg = Config.load_yaml(self.transform_cfg)
            self.opt["augmentations"] = self.transform_cfg
        else:
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

    def init_loading(self):
        self.weights = self.opt["global"].get("weights", None)
        if self.weights:
            state_dict = torch.load(self.weights, map_location="cpu")
            self.model = load_state_dict(self.model, state_dict, "model")

    def init_model(self):
        CLASSNAMES = getattr(self.dataset, "classnames", None)
        self.model = get_instance(
            self.opt["model"],
            registry=MODEL_REGISTRY,
            num_classes=len(CLASSNAMES) if CLASSNAMES is not None else None,
            classnames=CLASSNAMES,
        )
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
