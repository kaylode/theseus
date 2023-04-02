"""
Modified from https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/tools/program.py
"""

import json
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from copy import deepcopy

import yaml

from theseus.base.utilities.loading import load_yaml
from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


class Config(dict):
    """Single level attribute dict, recursive"""

    _depth = 0
    _yaml_paths = []

    # def __new__(class_, yaml_path, *args, **kwargs):
    #     if yaml_path in class_._yaml_paths:
    #         LOGGER.text(
    #             "Circular includes detected in YAML initialization!",
    #             level=LoggerObserver.CRITICAL,
    #         )
    #         raise ValueError()
    #     class_._yaml_paths.append(yaml_path)
    #     return dict.__new__(class_, yaml_path, *args, **kwargs)

    def __init__(self, yaml_path):
        super(Config, self).__init__()

        config = load_yaml(yaml_path)

        if "includes" in config.keys():
            final_config = {}
            for include_yaml_path in config["includes"]:
                tmp_config = Config(include_yaml_path)
                final_config.update(tmp_config)

            final_config.update(config)
            final_config.pop("includes")
            super(Config, self).update(final_config)
        else:
            super(Config, self).update(config)

        # self._yaml_paths.pop(-1)  # the last successful yaml will be popped out

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))

    def save_yaml(self, path):
        LOGGER.text(f"Saving config to {path}...", level=LoggerObserver.DEBUG)
        with open(path, "w") as f:
            yaml.dump(dict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, path):
        LOGGER.text(f"Loading config from {path}...", level=LoggerObserver.DEBUG)
        return cls(path)

    def __repr__(self) -> str:
        return str(json.dumps(dict(self), sort_keys=False, indent=4))


class Opts(ArgumentParser):
    def __init__(self):
        super(Opts, self).__init__(formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument(
            "-o", "--opt", nargs="+", help="override configuration options"
        )

    def parse_args(self, argv=None):
        args = super(Opts, self).parse_args(argv)
        assert args.config is not None, "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)

        config = Config(args.config)
        config = self.override(config, args.opt)
        return config

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            try:
                k, v = s.split("=")
            except ValueError:
                LOGGER.text(
                    "Invalid option: {}, options should be in the format of key=value".format(
                        s
                    ),
                    level=LoggerObserver.ERROR,
                )
                raise ValueError()

            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config

    def override(self, global_config, overriden):
        """
        Merge config into global config.
        Args:
            config (dict): Config to be merged.
        Returns: global config
        """
        LOGGER.text("Overriding configuration...", LoggerObserver.DEBUG)
        for key, value in overriden.items():
            if "." not in key:
                if isinstance(value, dict) and key in global_config:
                    global_config[key].update(value)
                else:
                    if key in global_config.keys():
                        global_config[key] = value
                    else:
                        LOGGER.text(
                            f"'{key}' not found in config",
                            level=LoggerObserver.WARN,
                        )
            else:
                sub_keys = key.split(".")
                assert (
                    sub_keys[0] in global_config
                ), "the sub_keys can only be one of global_config: {}, but get: {}, please check your running command".format(
                    global_config.keys(), sub_keys[0]
                )
                cur = global_config[sub_keys[0]]
                for idx, sub_key in enumerate(sub_keys[1:]):
                    if idx == len(sub_keys) - 2:
                        if sub_key in cur.keys():
                            cur[sub_key] = value
                        else:
                            LOGGER.text(
                                f"'{key}' not found in config",
                                level=LoggerObserver.WARN,
                            )
                    else:
                        cur = cur[sub_key]
        return global_config
