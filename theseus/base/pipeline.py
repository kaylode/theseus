from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset

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


class _PipelineBase:
    """
    Shared base for train/test pipelines, eliminating duplication of
    globals initialization, registry setup, and logging.
    """

    def __init__(self, opt: DictConfig) -> None:
        self.opt = opt
        self.seed = self.opt["global"].get("seed", 1702)
        seed_everything(self.seed)
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @initialized.setter
    def initialized(self, value: bool) -> None:
        self._initialized = value

    def _log(self, msg: str, level: int = LoggerObserver.INFO) -> None:
        """Convenience logging helper."""
        self.logger.text(msg, level=level)

    def _setup_savedir(self) -> str:
        """Create and return the experiment save directory."""
        exp_name = self.opt["global"].get("exp_name", None)
        exist_ok = self.opt["global"].get("exist_ok", False)
        save_dir = self.opt["global"].get("save_dir", "runs")

        if exp_name:
            savedir = os.path.join(save_dir, exp_name)
            if not exist_ok:
                savedir = get_new_folder_name(savedir)
        else:
            savedir = os.path.join(
                save_dir,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            )
        os.makedirs(savedir, exist_ok=True)
        return savedir

    def init_globals(self) -> None:
        """Initialize logger, experiment directory, and global variables."""
        self.logger = LoggerObserver.getLogger("main")

        # Global variables
        self.exp_name = self.opt["global"].get("exp_name", None)
        self.exist_ok = self.opt["global"].get("exist_ok", False)
        self.debug = self.opt["global"].get("debug", False)
        self.resume = self.opt["global"].get("resume", None)
        self.pretrained = self.opt["global"].get("pretrained", None)

        # Setup save directory
        self.savedir = self._setup_savedir()

        # File logging
        file_logger = FileLogger(__name__, self.savedir, debug=self.debug)
        self.logger.subscribe(file_logger)

        # Image logging
        image_logger = ImageWriter(self.savedir)
        self.logger.subscribe(image_logger)

        # Transform config
        self.transform_cfg = self.opt.get("augmentations", None)

        # Log config
        self._log("\n" + OmegaConf.to_yaml(self.opt))
        self._log(f"Everything will be saved to {self.savedir}")

    def init_registry(self) -> None:
        """Initialize component registries. Override in subclass to extend."""
        self.model_registry = MODEL_REGISTRY
        self.dataset_registry = DATASET_REGISTRY
        self.dataloader_registry = DATALOADER_REGISTRY
        self.metric_registry = METRIC_REGISTRY
        self.loss_registry = LOSS_REGISTRY
        self.callbacks_registry = CALLBACKS_REGISTRY
        self.trainer_registry = TRAINER_REGISTRY
        self.transform_registry = TRANSFORM_REGISTRY
        self._log(
            "You should override the init_registry() function",
            LoggerObserver.CRITICAL,
        )


class BasePipeline(_PipelineBase):
    """
    Full training pipeline that orchestrates all components:
    globals → registry → data → model → callbacks → trainer.

    Subclass this and override ``init_registry()`` to plug in task-specific
    registries.
    """

    def __init__(self, opt: DictConfig) -> None:
        super().__init__(opt)

    def _init_transforms(self) -> dict[str, Any]:
        """Initialize transforms, returning a dict with 'train'/'val' keys."""
        if self.transform_cfg is not None:
            return get_instance_recursively(self.transform_cfg, registry=self.transform_registry)
        return {"train": None, "val": None}

    def init_train_dataloader(self) -> None:
        self.transform = self._init_transforms()

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

        self._log(f"Number of training samples: {len(self.train_dataset)}")
        self._log(f"Number of training iterations each epoch: {len(self.train_dataloader)}")

    def init_validation_dataloader(self) -> None:
        self.transform = self._init_transforms()

        if self.opt["data"]["dataset"].get("val", None) is None:
            self._auto_split_dataset()
        else:
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
        self.classnames = getattr(self.val_dataset, "classnames", None)

        self._log(f"Number of validation samples: {len(self.val_dataset)}")
        self._log(f"Number of validation iterations each epoch: {len(self.val_dataloader)}")

    def _auto_split_dataset(self) -> None:
        """Auto-split training dataset when no validation set is provided."""
        split_ratio = self.opt.data.get("auto_split_ratio", 0.8)
        self._log(
            f"No validation dataset found. Auto splitting training dataset "
            f"with ratio={split_ratio}.",
            level=LoggerObserver.WARN,
        )
        train_size = len(self.train_dataset)
        val_size = int(train_size * (1 - split_ratio))

        train_dataset, val_dataset = (
            Subset(self.train_dataset, indices=indices)
            for indices in torch.split_with_sizes(
                torch.arange(train_size), [train_size - val_size, val_size]
            )
        )

        # Copy attributes from original dataset to subsets
        attrs = dir(self.train_dataset)
        for attr in attrs:
            if not attr.startswith("__"):
                setattr(train_dataset, attr, getattr(self.train_dataset, attr))
                setattr(val_dataset, attr, getattr(self.train_dataset, attr))

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_dataloader = get_instance_recursively(
            self.opt["data"]["dataloader"]["train"],
            registry=self.dataloader_registry,
            dataset=self.train_dataset,
        )

        self._log(f"Number of training samples: {len(self.train_dataset)}")
        self._log(f"Number of training iterations each epoch: {len(self.train_dataloader)}")

    def init_test_dataloader(self) -> None:
        """Initialize test dataset and dataloader, falling back to val if test is missing."""
        self.transform = get_instance_recursively(
            self.transform_cfg, registry=self.transform_registry
        )

        transform_cfg = (
            self.transform["test"] if "test" in self.transform else self.transform.get("val", None)
        )

        test_data_cfg = self.opt["data"]["dataset"].get("test", None)
        if test_data_cfg is None:
            self._log(
                "No test dataset found in config. Falling back to val dataset.",
                level=LoggerObserver.WARN,
            )
            test_data_cfg = self.opt["data"]["dataset"].get("val")

        self.test_dataset = get_instance_recursively(
            test_data_cfg,
            registry=self.dataset_registry,
            transform=transform_cfg,
        )

        test_loader_cfg = self.opt["data"]["dataloader"].get("test", None)
        if test_loader_cfg is None:
            self._log(
                "No test dataloader found in config. Falling back to val dataloader.",
                level=LoggerObserver.WARN,
            )
            test_loader_cfg = self.opt["data"]["dataloader"].get("val")

        self.test_dataloader = get_instance_recursively(
            test_loader_cfg,
            registry=self.dataloader_registry,
            dataset=self.test_dataset,
        )

        self.classnames = getattr(self.test_dataset, "classnames", None)

        self._log(f"Number of test samples: {len(self.test_dataset)}")
        self._log(f"Number of test iterations each epoch: {len(self.test_dataloader)}")

    def init_datamodule(self) -> None:
        self.datamodule = LightningDataModuleWrapper(
            trainloader=getattr(self, "train_dataloader", None),
            valloader=getattr(self, "val_dataloader", None),
            testloader=getattr(self, "test_dataloader", None),
        )

    def init_model(self) -> Any:
        CLASSNAMES = self.classnames
        model = get_instance(
            self.opt["model"],
            registry=self.model_registry,
            num_classes=len(CLASSNAMES) if CLASSNAMES is not None else None,
            classnames=CLASSNAMES,
        )
        return model

    def init_criterion(self) -> Any | None:
        CLASSNAMES = self.classnames
        if self.opt["loss"] is None:
            return None

        self.criterion = get_instance_recursively(
            self.opt["loss"],
            registry=self.loss_registry,
            num_classes=len(CLASSNAMES) if CLASSNAMES is not None else None,
            classnames=CLASSNAMES,
        )
        return self.criterion

    def init_model_with_loss(self, is_train: bool = True) -> None:
        self.model = self.init_model()
        criterion = self.init_criterion()
        num_epochs = self.opt["trainer"]["args"]["max_epochs"]
        batch_size = self.opt["data"]["dataloader"]["val"]["args"]["batch_size"]
        use_mixed_precision = self.opt["trainer"]["args"].get("precision", None)
        use_mixed_precision = bool(use_mixed_precision)

        self.model = LightningModelWrapper(
            self.model,
            criterion,
            use_mixed_precision=use_mixed_precision,
            datamodule=getattr(self, "datamodule", None),
            metrics=getattr(self, "metrics", None),
            optimizer_config=self.opt.get("optimizer", None) if is_train else None,
            scheduler_config=self.opt.get("scheduler", None) if is_train else None,
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
            state_dict = torch.load(pretrained, map_location="cpu", weights_only=False)
            try:
                self.model.load_state_dict(state_dict["state_dict"], strict=False)
                self._log(
                    f"Loaded pretrained model from {pretrained}",
                    level=LoggerObserver.SUCCESS,
                )
            except Exception as e:
                self._log(
                    f"Loaded pretrained model from {pretrained}. Mismatched keys: {e}",
                    level=LoggerObserver.WARN,
                )

    def init_metrics(self) -> None:
        CLASSNAMES = self.classnames
        if self.opt["metrics"] is None:
            self.metrics = None
            return

        self.metrics = get_instance_recursively(
            self.opt["metrics"],
            registry=self.metric_registry,
            num_classes=len(CLASSNAMES) if CLASSNAMES is not None else None,
            classnames=CLASSNAMES,
        )

    def init_callbacks(self) -> list[Any]:
        callbacks = get_instance_recursively(
            self.opt["callbacks"],
            save_dir=getattr(self, "savedir", "runs"),
            resume=getattr(self, "resume", None),
            config_dict=self.opt,
            registry=self.callbacks_registry,
        )
        return callbacks

    def init_trainer(self, callbacks: list[Any]) -> None:
        self.trainer = get_instance(
            self.opt["trainer"],
            default_root_dir=getattr(self, "savedir", "runs"),
            deterministic="warn",
            callbacks=callbacks,
            registry=self.trainer_registry,
        )

    def save_configs(self) -> None:
        with open(os.path.join(self.savedir, "pipeline.yaml"), "w") as f:
            OmegaConf.save(config=self.opt, f=f)

    def init_pipeline(self, phase: str = "train") -> None:
        if self.initialized:
            return
        self.init_globals()
        self.init_registry()

        if phase == "train":
            self.init_train_dataloader()
            self.init_validation_dataloader()
            if "test" in self.opt["data"]["dataset"]:
                self.init_test_dataloader()
            self.init_datamodule()
            self.init_metrics()
            self.init_model_with_loss()
            callbacks = self.init_callbacks()
            self.save_configs()
        else:
            if phase == "test":
                self.init_test_dataloader()
            else:
                self.init_validation_dataloader()
            self.init_datamodule()
            self.init_metrics()
            self.init_model_with_loss(is_train=False)
            callbacks = []

        # Always add core callbacks
        if getattr(self.model, "metrics", None):
            callbacks.insert(
                0,
                self.callbacks_registry.get("MetricLoggerCallback")(save_dir=self.savedir),
            )
        callbacks.insert(
            0,
            self.callbacks_registry.get("LossLoggerCallback")(
                print_interval=self.opt["trainer"]["args"].get("log_every_n_steps", None),
            ),
        )
        callbacks.insert(0, self.callbacks_registry.get("TimerCallback")())

        self.init_trainer(callbacks)
        self.initialized = True

    def fit(self) -> None:
        """Run the full training pipeline."""
        self.init_pipeline(phase="train")
        self.trainer.fit(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=self.resume,
        )

    def evaluate(self) -> dict[str, Any]:
        """Run validation and return metrics."""
        self.init_pipeline(phase="validation")
        self.trainer.validate(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=self.resume,
        )
        return self.trainer.callback_metrics

    def test(self) -> dict[str, Any]:
        """Run testing and return metrics."""
        self.init_pipeline(phase="test")
        self.trainer.test(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=self.resume,
        )
        return self.trainer.callback_metrics


class BaseTestPipeline(_PipelineBase):
    """
    Lightweight pipeline for inference/testing only.
    Shares globals/registry init logic with ``BasePipeline`` via ``_PipelineBase``.
    """

    def __init__(self, opt: DictConfig) -> None:
        super().__init__(opt)

    def init_globals(self) -> None:
        """Initialize globals without image writer (not needed for inference)."""
        self.logger = LoggerObserver.getLogger("main")

        self.exp_name = self.opt["global"].get("exp_name", None)
        self.exist_ok = self.opt["global"].get("exist_ok", False)
        self.debug = self.opt["global"].get("debug", False)
        self.transform_cfg = self.opt["global"].get("cfg_transform", None)

        self.savedir = self._setup_savedir()
        self.transform_cfg = self.opt.get("augmentations", None)

        file_logger = FileLogger(__name__, self.savedir, debug=self.debug)
        self.logger.subscribe(file_logger)
        self._log(str(self.opt))
        self._log(f"Everything will be saved to {self.savedir}")

    def init_registry(self) -> None:
        self.model_registry = MODEL_REGISTRY
        self.dataset_registry = DATASET_REGISTRY
        self.dataloader_registry = DATALOADER_REGISTRY
        self.transform_registry = TRANSFORM_REGISTRY
        self._log("You should override the init_registry() function")

    def init_test_dataloader(self) -> None:
        self.transform = get_instance_recursively(
            self.transform_cfg, registry=self.transform_registry
        )

        transform_cfg = (
            self.transform.get("test") if isinstance(self.transform, dict) else self.transform
        )
        if transform_cfg is None:
            transform_cfg = (
                self.transform.get("val") if isinstance(self.transform, dict) else self.transform
            )

        # Handle both nested (data.dataset.test) and flat (data.dataset) configs
        test_data_cfg = self.opt["data"]["dataset"]
        if "name" not in test_data_cfg:
            test_data_cfg = test_data_cfg.get("test") or test_data_cfg.get("val")

        self.dataset = get_instance(
            test_data_cfg,
            registry=DATASET_REGISTRY,
            transform=transform_cfg,
        )

        # Handle both nested (data.dataloader.test) and flat (data.dataloader) configs
        test_loader_cfg = self.opt["data"]["dataloader"]
        if "name" not in test_loader_cfg:
            test_loader_cfg = test_loader_cfg.get("test") or test_loader_cfg.get("val")

        self.dataloader = get_instance(
            test_loader_cfg,
            registry=DATALOADER_REGISTRY,
            dataset=self.dataset,
        )

        self._log(f"Number of test samples: {len(self.dataset)}")
        self._log(f"Number of test iterations each epoch: {len(self.dataloader)}")

    def init_model(self) -> None:
        CLASSNAMES = getattr(self.dataset, "classnames", None)
        self.model = get_instance(
            self.opt["model"],
            registry=MODEL_REGISTRY,
            num_classes=len(CLASSNAMES) if CLASSNAMES is not None else None,
            classnames=CLASSNAMES,
        )
        self.model = LightningModelWrapper(self.model)
        self.model.eval()

    def init_loading(self) -> None:
        self.weights = self.opt["global"].get("pretrained", None)
        if self.weights:
            state_dict = torch.load(self.weights, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state_dict["state_dict"], strict=False)

    def init_pipeline(self) -> None:
        self.init_globals()
        self.init_registry()
        self.init_test_dataloader()
        self.init_model()
        self.init_loading()

    def inference(self) -> Any:
        raise NotImplementedError()
