""" CUDA / AMP utils
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from typing import Any
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger('main')

def get_devices_info(device_names="0"):

    if device_names.startswith('cuda'):
        device_names = device_names.split('cuda:')[1]
    elif device_names.startswith('cpu'):
        return "CPU"

    devices_info = ""
    for i, device_id in enumerate(device_names.split(',')):
        p = torch.cuda.get_device_properties(i)
        devices_info += f"CUDA:{device_id} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    return devices_info

def get_device(name='cpu') -> torch.device:
    if name.startswith('cuda'):
        if not torch.cuda.is_available():
            LOGGER.text("CUDA is not available. Using CPU...", level=LoggerObserver.WARN)
            name = 'cpu'
    return torch.device(name)

def move_to(obj: Any, device: torch.device):
    """Credit: https://discuss.pytorch.org/t/pytorch-tensor-to-device-for-a-list-of-dict/66283
    Arguments:
        obj {dict, list} -- Object to be moved to device
        device {torch.device} -- Device that object will be moved to
    Raises:
        TypeError: object is of type that is not implemented to process
    Returns:
        type(obj) -- same object but moved to specified device
    """
    if torch.is_tensor(obj) or isinstance(obj, torch.nn.Module):
        return obj.to(device)
    if isinstance(obj, dict):
        res = {k: move_to(v, device) for k, v in obj.items()}
        return res
    if isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to(list(obj), device))
    
    return obj

def detach(obj: Any):
    """Credit: https://discuss.pytorch.org/t/pytorch-tensor-to-device-for-a-list-of-dict/66283
    Arguments:
        obj {dict, list} -- Object to be moved to cpu
    Raises:
        TypeError: Invalid type for detach
    Returns:
        type(obj) -- same object but moved to cpu
    """
    if torch.is_tensor(obj):
        return obj.detach()
    if isinstance(obj, dict):
        res = {k: detach(v) for k, v in obj.items()}
        return res
    if isinstance(obj, list):
        return [detach(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(detach(list(obj)))
    raise TypeError("Invalid type for detach")