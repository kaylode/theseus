import io
import os
import glob
import torch
import traceback
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from theseus.utilities.loggers.observer import LoggerObserver, LoggerSubscriber
LOGGER = LoggerObserver.getLogger('main')

class TensorboardLogger(LoggerSubscriber):
    """
    Logger for Tensorboard visualization
    :param log_dir: Path to save checkpoint
    """
    def __init__(self, log_dir, resume=None):
        self.log_dir = log_dir      
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Load old logging
        if resume is not None:
            self.load(resume)

    def log_scalar(self, tag, value, step, **kwargs):
        """
        Write a log to specified directory
        :param tags: (str) tag for log
        :param values: (number) value for corresponding tag
        :param step: (int) logging step
        """

        self.writer.add_scalar(tag, value, step)

    def log_figure(self, tag, value, step, **kwargs):
        """
        Write a matplotlib fig to tensorboard
        :param tags: (str) tag for log
        :param value: (image) image to log. torch.Tensor or plt.fire.Figure
        :param step: (int) logging step
        """

        if isinstance(value, torch.Tensor):
            self.writer.add_image(tag, value, global_step=step)
        else:
            self.writer.add_figure(tag, value, global_step=step)

    def log_torch_module(self, tag, value, inputs, **kwargs):
        """
        Write a model graph to tensorboard
        :param value: (nn.Module) torch model
        :param inputs: sample tensor
        """
        self.writer.add_graph(value, inputs)

    def log_embedding(self, tag, value, label_img=None, step=0, metadata=None, metadata_header=None, **kwargs):
        """
        Write a embedding projection to tensorboard
        :param value: (torch.Tensor) embedding (N, D)
        :param label_img: (torch.Tensor) normalized image tensors (N, 3, H, W)
        :param metadata: (List) list of coresponding labels
        """
        self.writer.add_embedding(
            tag=tag,
            mat=value, 
            label_img = label_img, 
            metadata=metadata, 
            metadata_header = metadata_header,
            global_step=step)

    def load(self, old_log):
        """
        Load tensorboard from log
        :param old_log: (str) path to previous log
        """
        all_logs, all_figs = tflog2pandas(old_log)

        for _, row in all_logs.iterrows():
            tag, value, step = row
            self.log_scalar(tag,value,step)

        for _, row in all_figs.iterrows():
            tag, value, step = row
            image_result = Image.open(io.BytesIO(value))
            image = ToTensor()(image_result)
            self.log_figure(tag, image, step)

    def __del__(self):
        self.writer.close()


def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    runfig_data = pd.DataFrame({"name": [], "value": [], "step": []})
    try:

        ## Scalar values
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])

        ## Image values
        tags = event_acc.Tags()["images"]
        for tag in tags:
            event_list = event_acc.Images(tag)
            values = list(map(lambda x: x.encoded_image_string, event_list))
            step = list(map(lambda x: x.step, event_list))

            r = {"name": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runfig_data = pd.concat([runfig_data, r])

    # Dirty catch of DataLossError
    except Exception:
        LOGGER.text("Event file possibly corrupt: {}".format(path), level=LoggerObserver.WARN)
        traceback.print_exc()
    return runlog_data, runfig_data

def find_old_log(weight_path):
    """
    Find log inside dir
    """
    pardir = os.path.dirname(weight_path)
    event_paths = glob.glob(os.path.join(pardir, "event*"))
    if len(event_paths) == 0:
        return None
    return event_paths[0]