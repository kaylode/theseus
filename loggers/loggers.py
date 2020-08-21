import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger():
    def __init__(self, log_dir='loggers/runs'):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.iters = {}
    def write(self, tags, values):
        if not isinstance(tags, list):
            tags = list(tags)
        if not isinstance(values, list):
            values = list(values)

        for i, (tag, value) in enumerate(zip(tags,values)):
            if tag not in self.iters.keys():
                self.iters[tag] = 0
            self.writer.add_scalar(tag, value, self.iters[tag])
            self.iters[tag] += 1


