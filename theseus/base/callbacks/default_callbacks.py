from typing import List, Dict
from theseus.base.callbacks.base_callbacks import Callbacks
from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

class DefaultCallbacks(Callbacks):
    """
    Default callbacks that will always be used
    """

    def on_start(self, logs: Dict=None):
        """
        Before going to the main loop
        """
        LOGGER.text(f'===========================START TRAINING=================================', level=LoggerObserver.INFO)

    def on_finish(self, logs: Dict=None):
        """
        After the main loop
        """
        LOGGER.text("Training Completed!", level=LoggerObserver.INFO)