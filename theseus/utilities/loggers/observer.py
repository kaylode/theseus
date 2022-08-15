import logging
import torch
import matplotlib as mpl

from typing import Dict, List
from .subscriber import LoggerSubscriber
from tabulate import tabulate
from inspect import getframeinfo, stack

def get_type(value):
    if isinstance(value, torch.nn.Module):
        return LoggerObserver.TORCH_MODULE
    if isinstance(value, mpl.figure.Figure):
        return LoggerObserver.FIGURE
    if isinstance(value, str):
        return LoggerObserver.TEXT
    if isinstance(value, torch.Tensor):
        if len(value.shape) == 2:
            return LoggerObserver.EMBED
    if isinstance(value, (int, float)):
        return LoggerObserver.SCALAR
    
    LoggerObserver.text(f'Fail to log undefined type: {type(value)}', level=LoggerObserver.CRITICAL)
    raise ValueError()

class LoggerObserver(object):
    """Logger Oberserver Degisn Pattern
    notifies every subscribers when .log() is called
    """
    SCALAR = 'scalar'
    FIGURE = 'figure'
    TORCH_MODULE = 'torch_module'
    TEXT = 'text'
    SPECIAL_TEXT = 'special_text'
    EMBED = 'embedding'
    TABLE = 'table'
    VIDEO = 'video'

    WARN = logging.WARN
    ERROR = logging.ERROR
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    CRITICAL = logging.CRITICAL
    SUCCESS = "SUCCESS"

    instances = {}

    def __new__(cls, name, *args, **kwargs):
        if name in LoggerObserver.instances.keys():
            return LoggerObserver.instances[name]
            
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, name) -> None:
        from .stdout_logger import StdoutLogger # to circumvent circular import

        self.subscriber = []
        self.name = name

        # Init with a stdout logger
        logger = StdoutLogger(__name__, debug=True)
        self.subscribe(logger)
        
        LoggerObserver.instances[name] = self

    @classmethod
    def getLogger(cls, name):
        if name in LoggerObserver.instances.keys():
            return LoggerObserver.instances[name]

        return cls(name)

    def subscribe(self, subscriber: LoggerSubscriber):
        self.subscriber.append(subscriber)

    def log(self, logs: List[Dict]):
        for subscriber in self.subscriber:
            for log in logs:
                tag = log['tag']
                value = log['value']
                log_type = log['type'] if 'type' in log.keys() else get_type(value)
                kwargs = log['kwargs'] if 'kwargs' in log.keys() else {}

                if log_type == LoggerObserver.SCALAR:
                    subscriber.log_scalar(
                        tag=tag,
                        value=value,
                        **kwargs
                    )

                if log_type == LoggerObserver.FIGURE:
                    subscriber.log_figure(
                        tag=tag,
                        value=value,
                        **kwargs
                    )

                if log_type == LoggerObserver.TORCH_MODULE:
                    subscriber.log_torch_module(
                        tag=tag,
                        value=value,
                        **kwargs
                    )

                if log_type == LoggerObserver.TEXT:
                    subscriber.log_text(
                        tag=tag,
                        value=value,
                        **kwargs
                    )

                if log_type == LoggerObserver.EMBED:
                    subscriber.log_embedding(
                        tag=tag,
                        value=value,
                        **kwargs
                    )

                if log_type == LoggerObserver.SPECIAL_TEXT:
                    subscriber.log_spec_text(
                        tag=tag,
                        value=value,
                        **kwargs
                    )

                if log_type == LoggerObserver.TABLE:
                    subscriber.log_table(
                        tag=tag,
                        value=value,
                        **kwargs
                    )

                if log_type == LoggerObserver.VIDEO:
                    subscriber.log_video(
                        tag=tag,
                        value=value,
                        **kwargs
                    )

    def text(self, value, level):
        """
        Text logging
        """
        caller = getframeinfo(stack()[1][0])
        function_name = stack()[1][3]
        filename = '//'.join(caller.filename.split('theseus')[1:])[1:] # split filename based on project name
        lineno = caller.lineno

        self.log([{
            'tag': 'stdout',
            'value': value,
            'type': LoggerObserver.TEXT,
            'kwargs': {
                'level': level,
                'lineno': lineno,
                'filename': filename,
                'funcname': function_name
            }
        }])

    def __repr__(self) -> str:
        table_headers = ["Subscribers"]
        table = tabulate(
            [[type(i).__name__] for i in self.subscriber], headers=table_headers, tablefmt="fancy_grid"
        )
        return "Logger subscribers: \n" + table