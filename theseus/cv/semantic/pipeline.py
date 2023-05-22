from theseus.base.pipeline import BasePipeline
from theseus.base.utilities.loggers import LoggerObserver
from theseus.cv.semantic.augmentations import TRANSFORM_REGISTRY
from theseus.cv.semantic.callbacks import CALLBACKS_REGISTRY
from theseus.cv.semantic.datasets import DATALOADER_REGISTRY, DATASET_REGISTRY
from theseus.cv.semantic.losses import LOSS_REGISTRY
from theseus.cv.semantic.metrics import METRIC_REGISTRY
from theseus.cv.semantic.models import MODEL_REGISTRY
from theseus.cv.semantic.trainer import TRAINER_REGISTRY
from omegaconf import DictConfig


class SemanticPipeline(BasePipeline):
    """docstring for Pipeline."""

    def __init__(self, opt: DictConfig):
        super(SemanticPipeline, self).__init__(opt)
        self.opt = opt

    def init_registry(self):
        super().init_registry()
        self.model_registry = MODEL_REGISTRY
        self.dataset_registry = DATASET_REGISTRY
        self.dataloader_registry = DATALOADER_REGISTRY
        self.metric_registry = METRIC_REGISTRY
        self.loss_registry = LOSS_REGISTRY
        self.callbacks_registry = CALLBACKS_REGISTRY
        self.trainer_registry = TRAINER_REGISTRY
        self.transform_registry = TRANSFORM_REGISTRY
        self.logger.text("Overidding registry in pipeline...", LoggerObserver.INFO)
