import inspect

from theseus.registry import Registry
from omegaconf import DictConfig, ListConfig

def get_instance_with_kwargs(registry, name, args: list = None, kwargs: dict = {}):
    # get keyword arguments from class signature
    inspection = inspect.signature(registry.get(name))
    class_kwargs = inspection.parameters.keys()

    if isinstance(args, (dict, DictConfig)):
        # override kwargs (from parent) with args (from config)
        kwargs.update(args)
        args = None

    if "kwargs" in class_kwargs:
        if args is None:
            return registry.get(name)(**kwargs)
        else:
            return registry.get(name)(*args, **kwargs)
    else:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in class_kwargs}
        if args is None:
            return registry.get(name)(**filtered_kwargs)
        else:
            return registry.get(name)(*args, **filtered_kwargs)


def get_instance(config, registry: Registry, **kwargs):
    # ref https://github.com/vltanh/torchan/blob/master/torchan/utils/getter.py
    assert "name" in config
    args = config.get("args", [])

    return get_instance_with_kwargs(registry, config["name"], args, kwargs)


def get_instance_recursively(config, registry: Registry, **kwargs):
    if isinstance(config, (list, tuple, ListConfig)):
        out = [
            get_instance_recursively(item, registry=registry, **kwargs)
            for item in config
        ]
        return out
    if isinstance(config, (dict, DictConfig)):
        if "name" in config.keys():
            if registry:
                args = get_instance_recursively(
                    config.get("args", {}), registry, **kwargs
                )
                return get_instance_with_kwargs(registry, config["name"], args, kwargs)

        else:
            out = {}
            for k, v in config.items():
                out[k] = get_instance_recursively(v, registry=registry, **kwargs)
            return out
    return config


def get_function(name):
    return globals()[name]
