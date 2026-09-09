from __future__ import annotations

import functools
import inspect
from typing import Any

from omegaconf import DictConfig, ListConfig

from theseus.registry import Registry


# Cache for inspect.signature to avoid repeated introspection
@functools.lru_cache(maxsize=256)
def _cached_signature(cls: type) -> inspect.Signature:
    """Cache inspect.signature results for performance."""
    return inspect.signature(cls)


def get_instance_with_kwargs(
    registry: Registry,
    name: str,
    args: Any = None,
    kwargs: dict[str, Any] | None = None,
) -> Any:
    """
    Instantiate a class from registry by name, filtering kwargs to match
    the class constructor signature.
    """
    if kwargs is None:
        kwargs = {}

    cls = registry.get(name)
    inspection = _cached_signature(cls)
    class_kwargs = inspection.parameters.keys()

    if isinstance(args, (dict, DictConfig)):
        # Override kwargs (from parent) with args (from config)
        kwargs.update(args)
        args = None

    if "kwargs" in class_kwargs:
        if args is None:
            return cls(**kwargs)
        else:
            return cls(*args, **kwargs)
    else:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in class_kwargs}
        if args is None:
            return cls(**filtered_kwargs)
        else:
            return cls(*args, **filtered_kwargs)


def get_instance(config: DictConfig | dict, registry: Registry, **kwargs: Any) -> Any:
    """
    Instantiate a single class from config dict with 'name' and optional 'args'.

    Args:
        config: Dict-like with 'name' key and optional 'args' key.
        registry: Registry to look up the class by name.
        **kwargs: Additional keyword arguments passed to the constructor.

    Returns:
        Instantiated object.
    """
    assert "name" in config, f"Config must contain 'name' key, got: {list(config.keys())}"
    args = config.get("args", [])
    return get_instance_with_kwargs(registry, config["name"], args, kwargs)


def get_instance_recursively(
    config: Any,
    registry: Registry,
    **kwargs: Any,
) -> Any:
    """
    Recursively walk a config tree and instantiate all objects that have
    a 'name' key, using the given registry.

    Supports nested lists, dicts, and DictConfig/ListConfig.
    """
    if isinstance(config, (list, tuple, ListConfig)):
        return [get_instance_recursively(item, registry=registry, **kwargs) for item in config]

    if isinstance(config, (dict, DictConfig)):
        if "name" in config:
            if registry:
                args = get_instance_recursively(config.get("args", {}), registry, **kwargs)
                return get_instance_with_kwargs(registry, config["name"], args, kwargs)
        else:
            return {
                k: get_instance_recursively(v, registry=registry, **kwargs)
                for k, v in config.items()
            }

    return config


def get_function(name: str) -> Any:
    """Get a function by name from the global scope."""
    return globals()[name]
