from abc import ABC, abstractmethod
from typing import Any


class Metric(ABC):
    """
    Abstract base class for all metrics in Theseus.

    Subclasses must implement ``update()``, ``value()``, ``reset()``,
    ``summary()``, and ``__str__()``.

    Example::

        @METRIC_REGISTRY.register()
        class MyAccuracy(Metric):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.correct = 0
                self.total = 0

            def update(self, outputs, batch):
                ...

            def value(self):
                return {"accuracy": self.correct / max(self.total, 1)}

            def reset(self):
                self.correct = 0
                self.total = 0
    """

    def __init__(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def update(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        """Update metric state with new predictions and targets."""
        ...

    @abstractmethod
    def value(self) -> dict[str, Any]:
        """Compute and return the metric value(s) as a dict."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset metric state for next epoch."""
        ...

    def summary(self) -> str:
        """Return a human-readable summary of the metric."""
        return str(self.value())

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.summary()}"
