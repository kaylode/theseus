import os
from datetime import datetime

from omegaconf import DictConfig, OmegaConf

from theseus.base.utilities.folder import get_new_folder_name
from theseus.base.utilities.getter import get_instance, get_instance_recursively
from theseus.base.utilities.loggers import FileLogger, ImageWriter, LoggerObserver
from theseus.base.utilities.seed import seed_everything
from theseus.ml.callbacks import CALLBACKS_REGISTRY
from theseus.ml.datasets import DATALOADER_REGISTRY, DATASET_REGISTRY
from theseus.ml.metrics import METRIC_REGISTRY
from theseus.ml.models import MODEL_REGISTRY
from theseus.ml.preprocessors import TRANSFORM_REGISTRY
from theseus.ml.trainer import TRAINER_REGISTRY


class MLPipeline(object):
    """docstring for Pipeline."""

    def __init__(self, opt: DictConfig):
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

    def init_registry(self):
        self.callbacks_registry = CALLBACKS_REGISTRY
        self.transform_registry = TRANSFORM_REGISTRY
        self.model_registry = MODEL_REGISTRY
        self.metric_registry = METRIC_REGISTRY
        self.trainer_registry = TRAINER_REGISTRY
        self.dataset_registry = DATASET_REGISTRY
        self.dataloader_registry = DATALOADER_REGISTRY
        self.logger.text("Overidding registry in pipeline...", LoggerObserver.INFO)

    def init_model(self):
        classnames = self.val_dataset["classnames"]
        num_classes = len(classnames)
        self.model = get_instance(
            self.opt["model"], num_classes=num_classes, registry=self.model_registry
        )

    def init_train_dataloader(self):
        self.transform = get_instance_recursively(
            self.transform_cfg, registry=self.transform_registry
        )
        self.train_dataset = get_instance_recursively(
            self.opt["data"]["dataset"]["train"],
            registry=self.dataset_registry,
            transform=self.transform["train"],
        ).load_data()

        self.logger.text(
            f"Training shape: {self.train_dataset['inputs'].shape}",
            level=LoggerObserver.INFO,
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

    def init_validation_dataloader(self):
        self.transform = get_instance_recursively(
            self.transform_cfg, registry=self.transform_registry
        )
        self.val_dataset = get_instance_recursively(
            self.opt["data"]["dataset"]["val"],
            registry=self.dataset_registry,
            transform=self.transform["val"],
        ).load_data()

        classnames = self.val_dataset["classnames"]
        num_classes = len(classnames)

        self.logger.text(
            f"Validation shape: {self.val_dataset['inputs'].shape}",
            level=LoggerObserver.INFO,
        )
        self.logger.text(
            f"Number of classes: {num_classes}",
            level=LoggerObserver.INFO,
        )

    def init_trainer(self, callbacks=None):
        self.trainer = get_instance(
            self.opt["trainer"],
            model=self.model,
            trainset=getattr(self, "train_dataset", None),
            valset=getattr(self, "val_dataset", None),
            metrics=self.metrics,
            callbacks=callbacks,
            registry=self.trainer_registry,
        )

    def init_loading(self):
        if getattr(self, "pretrained", None):
            self.model.load_model(self.pretrained)

    def init_pipeline(self, train=False):
        if self.initialized:
            return
        self.init_globals()
        self.init_registry()
        if train:
            self.init_train_dataloader()
            self.init_validation_dataloader()
            self.init_model()
            self.init_loading()
            self.init_metrics()
            callbacks = self.init_callbacks()
            self.save_configs()
        else:
            self.init_validation_dataloader()
            self.init_model()
            self.init_metrics()
            self.init_loading()
            callbacks = []

        self.init_trainer(callbacks=callbacks)
        self.initialized = True

    def save_configs(self):
        with open(os.path.join(self.savedir, "pipeline.yaml"), "w") as f:
            OmegaConf.save(config=self.opt, f=f)

    def fit(self):
        self.init_pipeline(train=True)
        self.trainer.fit()

    def evaluate(self):
        self.init_pipeline(train=False)
        return self.trainer.validate()
