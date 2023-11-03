from typing import *

import pandas as pd
import pandas_profiling as pp

from theseus.base.utilities.loggers.observer import LoggerObserver
from theseus.ml.callbacks import Callbacks

LOGGER = LoggerObserver.getLogger("main")


class PandasProfilerCallbacks(Callbacks):
    """
    Callbacks for making data profile
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()

    def on_sanity_check_start(self, logs: Dict) -> None:
        """
        Sanitycheck before starting. Run only when debug=True
        """
        import pdb

        pdb.set_trace()
        trainloader = logs["trainer"].train_dataloader
        iters = trainer.global_step
        model = pl_module.model
        valloader = pl_module.datamodule.valloader
        trainloader = pl_module.datamodule.trainloader
        train_batch = next(iter(trainloader))
        val_batch = next(iter(valloader))

        profile = pp.ProfileReport(data)
        profile.to_file("output.html")

        try:
            self.visualize_model(model, train_batch)
        except TypeError as e:
            LOGGER.text("Cannot log model architecture", level=LoggerObserver.ERROR)
        self.visualize_gt(train_batch, val_batch, iters)
