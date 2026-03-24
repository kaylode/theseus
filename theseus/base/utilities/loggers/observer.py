import logging
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import torch

mpl.use("Agg")

import threading
from inspect import getframeinfo, stack
from typing import Any, Callable, Dict, List, Optional

from tabulate import tabulate

from .subscriber import LoggerSubscriber


def get_type(value: Any) -> str:
    """Infer the log type from a value."""
    if isinstance(value, torch.nn.Module):
        return LoggerObserver.TORCH_MODULE
    if isinstance(value, (mpl.figure.Figure, go.Figure)):
        return LoggerObserver.FIGURE
    if isinstance(value, (torch.Tensor, np.ndarray)):
        if len(value.shape) == 2:
            return LoggerObserver.EMBED
    if isinstance(value, (int, float)):
        return LoggerObserver.SCALAR
    if isinstance(value, str):
        if value.endswith(".html"):
            return LoggerObserver.HTML
        else:
            return LoggerObserver.TEXT
    raise ValueError(f"Fail to log undefined type: {type(value)}")


class LoggerObserver:
    """
    Logger Observer Design Pattern.

    Notifies every subscriber when ``.log()`` is called.
    Uses a dispatch table instead of if-chains for O(1) routing.

    Example::

        logger = LoggerObserver.getLogger("main")
        logger.text("Hello world", level=LoggerObserver.INFO)
        logger.log([{"tag": "loss", "value": 0.5, "type": "scalar"}])
    """

    # Log type constants
    SCALAR = "scalar"
    FIGURE = "figure"
    TORCH_MODULE = "torch_module"
    TEXT = "text"
    SPECIAL_TEXT = "special_text"
    EMBED = "embedding"
    TABLE = "table"
    VIDEO = "video"
    HTML = "html"

    # Log level constants
    WARN = logging.WARN
    ERROR = logging.ERROR
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    CRITICAL = logging.CRITICAL
    SUCCESS = "SUCCESS"

    # Singleton instances
    instances: Dict[str, "LoggerObserver"] = {}
    _lock = threading.Lock()

    # Dispatch table: maps log type -> subscriber method name
    _DISPATCH: Dict[str, str] = {
        SCALAR: "log_scalar",
        FIGURE: "log_figure",
        TORCH_MODULE: "log_torch_module",
        TEXT: "log_text",
        SPECIAL_TEXT: "log_spec_text",
        EMBED: "log_embedding",
        TABLE: "log_table",
        VIDEO: "log_video",
        HTML: "log_html",
    }

    def __new__(cls, name: Optional[str] = None, *args: Any, **kwargs: Any) -> "LoggerObserver":
        with cls._lock:
            if name is None:
                name = str(os.getpid())
            if name in LoggerObserver.instances:
                return LoggerObserver.instances[name]
            return object.__new__(cls)

    def __init__(self, name: str) -> None:
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.subscriber: List[LoggerSubscriber] = []
        self.name = name

        # Init with a stdout logger
        from .stdout_logger import StdoutLogger

        logger = StdoutLogger(name=self.name, debug=True)
        self.subscribe(logger)

        LoggerObserver.instances[name] = self

    def __del__(self) -> None:
        for subscriber in self.subscriber:
            del subscriber
        if self.name in LoggerObserver.instances:
            LoggerObserver.instances.pop(self.name, None)

    @classmethod
    def getLogger(cls, name: str) -> "LoggerObserver":
        """Get or create a logger by name."""
        if name in LoggerObserver.instances:
            return LoggerObserver.instances[name]
        return cls(name)

    def subscribe(self, subscriber: LoggerSubscriber) -> None:
        """Add a subscriber that will receive log events."""
        self.subscriber.append(subscriber)

    def log(self, logs: List[Dict[str, Any]]) -> None:
        """
        Dispatch log entries to all subscribers using the dispatch table.
        Each log entry must have 'tag' and 'value' keys, with optional 'type' and 'kwargs'.
        """
        for subscriber in self.subscriber:
            for entry in logs:
                tag = entry["tag"]
                value = entry["value"]
                log_type = entry.get("type", get_type(value))
                kwargs = entry.get("kwargs", {})

                # Use dispatch table for O(1) routing
                method_name = self._DISPATCH.get(log_type)
                if method_name is not None:
                    method = getattr(subscriber, method_name, None)
                    if method is not None:
                        method(tag=tag, value=value, **kwargs)

    def text(self, value: Any, level: int = logging.INFO) -> None:
        """Convenience method for text logging with source location."""
        caller = getframeinfo(stack()[1][0])
        function_name = stack()[1][3]
        filename = "//".join(caller.filename.split("theseus")[1:])[
            1:
        ]  # split filename based on project name
        lineno = caller.lineno

        self.log(
            [
                {
                    "tag": "stdout",
                    "value": value,
                    "type": LoggerObserver.TEXT,
                    "kwargs": {
                        "level": level,
                        "lineno": lineno,
                        "filename": filename,
                        "funcname": function_name,
                    },
                }
            ]
        )

    def __repr__(self) -> str:
        table_headers = ["Subscribers"]
        table = tabulate(
            [[type(i).__name__] for i in self.subscriber],
            headers=table_headers,
            tablefmt="fancy_grid",
        )
        return "Logger subscribers: \n" + table
