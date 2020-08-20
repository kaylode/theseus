import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter







class Logger():
    def __init__(self, log_dir='loggers/runs'):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def log(self, text):
        for n_iter in range(100):
            self.writer.add_scalar('Loss/train', np.random.random(), n_iter)
            self.writer.add_scalar('Loss/test', np.random.random(), n_iter)
            self.writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
            self.writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
        #self.writer.close()

