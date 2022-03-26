import os
import yaml
import torch
import glob
import logging
from theseus.base.optimizers.scalers.native import NativeScaler
from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")

def load_yaml(path):
    with open(path, 'rt') as f:
        return yaml.safe_load(f)

def load_state_dict(instance, state_dict, key=None):
    """
    Load trained model checkpoint
    :param model: (nn.Module)
    :param path: (string) checkpoint path
    """

    if isinstance(instance, torch.nn.Module) or isinstance(instance, torch.optim.Optimizer) or isinstance(instance, NativeScaler):
        try:
            if key is not None:
                instance.load_state_dict(state_dict[key])
            else:
                instance.load_state_dict(state_dict)

            LOGGER.text("Loaded Successfully!", level=LoggerObserver.SUCCESS)
        except RuntimeError as e:
            LOGGER.text(f'Loaded Successfully. Ignoring {e}', level=LoggerObserver.WARN)
        return instance
    else:
        if key in state_dict.keys():    
            return state_dict[key]
        else:
            LOGGER.text(f"Cannot load key={key} from state_dict", LoggerObserver.WARN)

def find_old_tflog(pardir):
    event_paths = glob.glob(os.path.join(pardir, "event*"))
    if len(event_paths) == 0:
        return None
    else:
        return event_paths[0]
