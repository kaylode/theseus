from typing import Dict, List

from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


class MetricLoggerCallbacks(Callbacks):
    """
    Callbacks for logging running metric while training every epoch end
    Features:
        - Only do logging

    print_interval: `int`
        iteration cycle to log out
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def on_val_epoch_end(self, logs: Dict = None):
        """
        After finish validation
        """

        iters = logs["iters"]
        metric_dict = logs["metric_dict"]

        # Log metric
        metric_string = ""
        for metric, score in metric_dict.items():
            if isinstance(score, (int, float)):
                metric_string += metric + ": " + f"{score:.5f}" + " | "
        metric_string += "\n"

        LOGGER.text(metric_string, level=LoggerObserver.INFO)

        # Call other loggers
        log_dict = [
            {"tag": f"Validation/{k}", "value": v, "kwargs": {"step": iters}}
            for k, v in metric_dict.items()
        ]

        LOGGER.log(log_dict)
