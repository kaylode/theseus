from theseus.opt import Config
from theseus.base.pipeline import BasePipeline
from theseus.tabular.losses import LOSS_REGISTRY
from theseus.tabular.models import MODEL_REGISTRY
from theseus.utilities.loggers import LoggerObserver


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
        self.loss_registry = LOSS_REGISTRY
        self.logger.text(
            "Overidding registry in pipeline...", LoggerObserver.INFO
        )
