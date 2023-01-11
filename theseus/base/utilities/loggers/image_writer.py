import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from .observer import LoggerObserver
from .subscriber import LoggerSubscriber

LOGGER = LoggerObserver.getLogger("main")

mpl.use("Agg")


class ImageWriter(LoggerSubscriber):
    """Logger for writing images"""

    def __init__(self, savedir) -> None:
        self.savedir = savedir

    def log_figure(self, tag: str, value, **kwargs):
        savepath = os.path.join(self.savedir, tag)
        dirname = os.path.dirname(savepath)
        os.makedirs(dirname, exist_ok=True)

        if isinstance(value, go.Figure):
            value.write_image(savepath + ".png")
        if isinstance(value, mpl.figure.Figure):
            value.savefig(savepath)

        LOGGER.text(f"Saved image to {savepath}", level=LoggerObserver.INFO)
