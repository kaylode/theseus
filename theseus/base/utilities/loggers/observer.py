import logging
import os
import sys
import threading
from inspect import getframeinfo, stack
from typing import Any

import numpy as np
from tabulate import tabulate

from .subscriber import LoggerSubscriber


def get_type(value: Any) -> str:
    """Infer the log type from a value."""
    if "torch" in sys.modules:
        import torch

        if isinstance(value, torch.nn.Module):
            return LoggerObserver.TORCH_MODULE
        if isinstance(value, torch.Tensor) and len(value.shape) == 2:
            return LoggerObserver.EMBED

    if "matplotlib" in sys.modules:
        import matplotlib as mpl

        if isinstance(value, mpl.figure.Figure):
            return LoggerObserver.FIGURE

    if "plotly" in sys.modules:
        import plotly.graph_objs as go

        if isinstance(value, go.Figure):
            return LoggerObserver.FIGURE

    if isinstance(value, np.ndarray) and len(value.shape) == 2:
        return LoggerObserver.EMBED

    if isinstance(value, (int, float, np.number)):
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
    instances: dict[str, "LoggerObserver"] = {}
    _lock = threading.Lock()

    # Dispatch table: maps log type -> subscriber method name
    _DISPATCH: dict[str, str] = {
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

    def __new__(cls, name: str | None = None, *args: Any, **kwargs: Any) -> "LoggerObserver":
        if name is None:
            name = str(os.getpid())
        with cls._lock:
            if name not in cls.instances:
                instance = object.__new__(cls)
                instance._initialized = False
                cls.instances[name] = instance
            return cls.instances[name]

    def __init__(self, name: str | None = None) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.subscriber: list[LoggerSubscriber] = []
        if name is None:
            name = str(os.getpid())
        self.name = name

        # Init with a stdout logger
        from .stdout_logger import StdoutLogger

        logger = StdoutLogger(name=self.name, debug=True)
        self.subscribe(logger)

    def __del__(self) -> None:
        for subscriber in self.subscriber:
            del subscriber
        if self.name in LoggerObserver.instances:
            LoggerObserver.instances.pop(self.name, None)

    @classmethod
    def getLogger(cls, name: str) -> "LoggerObserver":
        """Get or create a logger by name."""
        return cls(name)

    def subscribe(self, subscriber: LoggerSubscriber) -> None:
        """Add a subscriber that will receive log events."""
        self.subscriber.append(subscriber)

    def log(self, logs: list[dict[str, Any]]) -> None:
        """
        Dispatch log entries to all subscribers using the dispatch table.
        Each log entry must have 'tag' and 'value' keys, with optional 'type' and 'kwargs'.
        """
        # Support distributed logging
        is_master = True
        if "torch" in sys.modules:
            import torch

            if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
                is_master = False
        if not is_master:
            return

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

    def text(self, *value: Any, level: int = logging.INFO) -> None:
        """Convenience method for text logging with source location."""
        caller = getframeinfo(stack()[1][0])
        function_name = stack()[1][3]
        filename = "//".join(caller.filename.split("theseus")[1:])[
            1:
        ]  # split filename based on project name
        lineno = caller.lineno

        texts = []
        for v in value:
            if isinstance(v, dict):
                import json

                texts.append(json.dumps(v, indent=4, default=str))
            else:
                texts.append(str(v))
        value_str = " ".join(texts)

        self.log(
            [
                {
                    "tag": "stdout",
                    "value": value_str,
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
