from typing import Dict

from theseus.base.callbacks import Callbacks
from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


class DebugCallbacks(Callbacks):
    """
    Callbacks for debugging.
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()

    def sanitycheck(self, logs: Dict = None):
        """
        Sanitycheck before starting. Run only when debug=True
        """

        LOGGER.text("Start sanity checks", level=LoggerObserver.DEBUG)
        self.params["trainer"].evaluate_epoch()
