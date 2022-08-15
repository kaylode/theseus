import os
import yaml
import torch
import glob
from theseus.base.optimizers.scalers.native import NativeScaler
from theseus.utilities.loggers.observer import LoggerObserver
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

LOGGER = LoggerObserver.getLogger("main")

def load_yaml(path):
    with open(path, 'rt') as f:
        return yaml.safe_load(f)

def load_state_dict(instance, state_dict, key=None, strict=True):
    """
    Load trained model checkpoint
    :param model: (nn.Module)
    :param path: (string) checkpoint path
    """
    
    if isinstance(instance, (
        _LRScheduler, ReduceLROnPlateau, 
        torch.nn.Module, torch.optim.Optimizer, 
        NativeScaler)) or getattr(instance, 'load_state_dict', None) is not None:
        try:
            if key is not None:
                _state_dict = state_dict[key]
            else:
                _state_dict = state_dict

            if not strict:
                instance.load_state_dict(_state_dict, strict=False)
            else:
                instance.load_state_dict(_state_dict)

            LOGGER.text("Loaded Successfully!", level=LoggerObserver.SUCCESS)
        except RuntimeError as e:
            if not strict:
                LOGGER.text(f'Loaded Successfully. Ignoring {e}', level=LoggerObserver.WARN)
            else:
                LOGGER.text(f'Loaded failed: "{e}". Consider loading with strict=False', level=LoggerObserver.ERROR)
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