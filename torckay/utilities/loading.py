import os
import yaml
import torch
import glob
import logging
from torckay.base.optimizers.scalers.native import NativeScaler
from torckay.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

def load_yaml(path):
    with open(path, 'rt') as f:
        return yaml.safe_load(f)

def load_state_dict(instance, state_dict, key):
    """
    Load trained model checkpoint
    :param model: (nn.Module)
    :param path: (string) checkpoint path
    """

    if isinstance(instance, torch.nn.Module) or isinstance(instance, torch.optim.Optimizer) or isinstance(instance, NativeScaler):
        try:
            instance.load_state_dict(state_dict[key])
            LOGGER.text("Loaded Successfully!", level=LoggerObserver.INFO)
        except RuntimeError as e:
            LOGGER.text(f'[Warning] Ignoring {e}', level=LoggerObserver.WARN)
        return instance
    else:
        return state_dict[key]

def find_old_tflog(pardir):
    event_paths = glob.glob(os.path.join(pardir, "event*"))
    if len(event_paths) == 0:
        return None
    else:
        return event_paths[0]