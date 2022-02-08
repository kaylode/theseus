""" CUDA / AMP utils
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch

def get_devices_info(gpu_devices="0"):
    devices_info = ""
    for i, device_id in enumerate(gpu_devices.split(',')):
        p = torch.cuda.get_device_properties(i)
        devices_info += f"CUDA:{device_id} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    return devices_info

