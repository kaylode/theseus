from theseus.opt import Config
from theseus.base.pipeline import BasePipeline
from theseus.cv.semantic.augmentations import TRANSFORM_REGISTRY
from theseus.cv.semantic.losses import LOSS_REGISTRY
from theseus.cv.semantic.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from theseus.cv.semantic.trainer import TRAINER_REGISTRY
from theseus.cv.semantic.metrics import METRIC_REGISTRY
from theseus.cv.semantic.models import MODEL_REGISTRY
from theseus.cv.semantic.callbacks import CALLBACKS_REGISTRY
from theseus.base.utilities.loggers import LoggerObserver


class Pipeline(BasePipeline):
    """docstring for Pipeline."""

    def __init__(
        self,
        opt: Config
    ):
        super(Pipeline, self).__init__(opt)
        self.opt = opt

    def init_registry(self):
        super().init_registry()
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
            "Overidding registry in pipeline...", LoggerObserver.INFO
        )
