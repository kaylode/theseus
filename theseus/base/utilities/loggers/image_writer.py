import os
from .subscriber import LoggerSubscriber

class ImageWriter(LoggerSubscriber):
    """Logger for writing images
    
    """
    def __init__(self, savedir) -> None:
        self.savedir = savedir

    def log_figure(self, tag: str, value, **kwargs):
        savepath = os.path.join(self.savedir, tag)
        dirname = os.path.dirname(savepath)
        os.makedirs(dirname, exist_ok=True)
        value.savefig(savepath)
