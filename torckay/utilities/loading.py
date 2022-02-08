import yaml
import torch

from torckay.utilities.loggers.logger import LoggerManager
LOGGER = LoggerManager.init_logger(__name__)

def load_yaml(path):
    with open(path, 'rt') as f:
        return yaml.safe_load(f)

def load_state_dict(instance, state_dict, key):
    """
    Load trained model checkpoint
    :param model: (nn.Module)
    :param path: (string) checkpoint path
    """

    if isinstance(instance, torch.nn.Module) or isinstance(instance, torch.optim.Optimizer):
        try:
            instance.load_state_dict(state_dict[key])
        except RuntimeError as e:
            LOGGER.warn(f'[Warning] Ignoring {e}')
    else:
        return state_dict[key]
    LOGGER.info("Loaded Successfully!")