import os
import glob
import traceback
import pandas as pd
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

class Logger():
    """
    Logger for Tensorboard visualization
    :param log_dir: Path to save checkpoint
    """
    def __init__(self, log_dir=None, resume=None):
        self.log_dir = log_dir
        
        if self.log_dir is None:
            self.log_dir = os.path.join('./loggers/runs',datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.datetime = os.path.basename(self.log_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Load old logging
        if resume is not None:
            self.load(resume)

    def write(self, tags, values, step):
        """
        Write a log to specified directory
        :param tags: (str) tag for log
        :param values: (number) value for corresponding tag
        :param step: (int) logging step
        """
        if not isinstance(tags, list):
            tags = list(tags)
        if not isinstance(values, list):
            values = list(values)

        for i, (tag, value) in enumerate(zip(tags,values)):
            self.writer.add_scalar(tag, value, step)

    def write_image(self, tag, image, step):
        """
        Write a matplotlib fig to tensorboard
        :param tags: (str) tag for log
        :param image: (image) image to log
        :param step: (int) logging step
        """

        self.writer.add_figure(tag, image, global_step=step)

    def write_text(self, tag, text, step):
        """
        Write a text to tensorboard
        :param tags: (str) tag for log
        :param text: (str) text to log
        :param step: (int) logging step
        """

        self.writer.add_text(tag, text, global_step=step)

    def load(self, old_log):
        """
        Load tensorboard from log
        :param old_log: (str) path to previous log
        """
        all_log = tflog2pandas(old_log)

        for _, row in all_log.iterrows():
            tag, value, step = row
            self.writer.add_scalar(tag,value,step)


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
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        # tags = event_acc.Tags()["images"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data

def find_old_log(weight_path):
    """
    Find log inside dir
    """
    pardir = os.path.dirname(weight_path)
    event_paths = glob.glob(os.path.join(pardir, "event*"))
    if len(event_paths) == 0:
        return None
    else:
        return event_paths[0]