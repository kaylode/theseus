from theseus.base.pipeline import BasePipeline
from theseus.base.utilities.getter import get_instance, get_instance_recursively
from theseus.base.utilities.loggers import LoggerObserver
from theseus.opt import Config
from theseus.tabular.base.preprocessors import TRANSFORM_REGISTRY
from theseus.tabular.classification.callbacks import CALLBACKS_REGISTRY
from theseus.tabular.classification.datasets import (
    DATALOADER_REGISTRY,
    DATASET_REGISTRY,
)
from theseus.tabular.classification.metrics import METRIC_REGISTRY
from theseus.tabular.classification.models import MODEL_REGISTRY
from theseus.tabular.classification.trainer import TRAINER_REGISTRY


class TabularPipeline(BasePipeline):
    """docstring for Pipeline."""

    def __init__(self, opt: Config):
        super(TabularPipeline, self).__init__(opt)
        self.opt = opt

    def init_registry(self):
        super().init_registry()
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

    def init_datasets(self):

        self.transform = get_instance_recursively(
            self.transform_cfg, registry=self.transform_registry
        )
        self.train_dataset = get_instance_recursively(
            self.opt["data"]["dataset"]["train"],
            registry=self.dataset_registry,
            transform=self.transform["train"],
        ).load_data()

        self.val_dataset = get_instance_recursively(
            self.opt["data"]["dataset"]["val"],
            registry=self.dataset_registry,
            transform=self.transform["val"],
        ).load_data()

        classnames = self.val_dataset["classnames"]
        num_classes = len(classnames)

        self.logger.text(
            f"Training shape: {self.train_dataset['inputs'].shape}",
            level=LoggerObserver.INFO,
        )
        self.logger.text(
            f"Validation shape: {self.val_dataset['inputs'].shape}",
            level=LoggerObserver.INFO,
        )
        self.logger.text(
            f"Number of classes: {num_classes}",
            level=LoggerObserver.INFO,
        )

    def init_metrics(self):
        classnames = self.val_dataset["classnames"]
        num_classes = len(classnames)
        self.metrics = get_instance_recursively(
            self.opt["metrics"],
            num_classes=num_classes,
            classnames=classnames,
            registry=self.metric_registry,
        )

    def init_callbacks(self):
        callbacks = get_instance_recursively(
            self.opt["callbacks"],
            save_dir=getattr(self, "save_dir", "runs"),
            resume=getattr(self, "resume", None),
            config_dict=self.opt,
            registry=self.callbacks_registry,
        )
        return callbacks

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

    def init_pipeline(self):
        self.init_globals()
        self.init_registry()
        self.init_datasets()
        self.init_model()
        self.init_metrics()
        callbacks = self.init_callbacks()
        self.init_trainer(callbacks=callbacks)

    def fit(self):
        self.init_pipeline()
        self.trainer.fit()
