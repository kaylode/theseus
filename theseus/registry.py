# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Enhanced for Theseus v2.0
from __future__ import annotations

import logging
from typing import Any, Generic, Iterator, Optional, TypeVar, overload

from tabulate import tabulate

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)

    To merge registries:

    .. code-block:: python
        MERGED = BACKBONE_REGISTRY.merge(OTHER_REGISTRY)
    """

    __slots__ = ("_name", "_obj_map")

    def __init__(self, name: str) -> None:
        self._name: str = name
        self._obj_map: dict[str, T] = {}

    def _do_register(self, name: str, obj: T, override: bool = False) -> None:
        if name in self._obj_map and self._obj_map[name] is not obj:
            if not override:
                logger.warning(
                    "An object named '%s' was already registered in '%s' registry!",
                    name,
                    self._name,
                )
                return
        self._obj_map[name] = obj

    @overload
    def register(self, obj: None = None, prefix: str = "", override: bool = False) -> Any: ...

    @overload
    def register(self, obj: T, prefix: str = "", override: bool = False) -> None: ...

    def register(
        self, obj: T | None = None, prefix: str = "", override: bool = False
    ) -> Any:
        """
        Register the given object under the name ``obj.__name__``.
        Can be used as either a decorator or not.
        """
        if obj is None:
            # Used as a decorator
            def deco(func_or_class: Any) -> Any:
                name = func_or_class.__name__
                self._do_register(prefix + name, func_or_class, override)
                return func_or_class

            return deco

        # Used as a function call
        name = obj.__name__  # type: ignore[union-attr]
        self._do_register(prefix + name, obj, override)

    def get(self, name: str) -> T:
        """Get registered object by name. Raises KeyError if not found."""
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                f"No object named '{name}' found in '{self._name}' registry! "
                f"Available: {sorted(self._obj_map.keys())}"
            )
        return ret

    def get_or_none(self, name: str) -> T | None:
        """Get registered object by name, returns None if not found."""
        return self._obj_map.get(name)

    def keys(self) -> list[str]:
        """Return all registered names."""
        return list(self._obj_map.keys())

    def values(self) -> list[T]:
        """Return all registered objects."""
        return list(self._obj_map.values())

    def merge(self, other: Registry[T], override: bool = False) -> Registry[T]:
        """
        Merge another registry into this one. Returns self for chaining.
        Useful for task-specific pipelines that extend base registries.
        """
        for name, obj in other:
            self._do_register(name, obj, override=override)
        return self

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __len__(self) -> int:
        return len(self._obj_map)

    def __getitem__(self, name: str) -> T:
        return self.get(name)

    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        table = tabulate(
            self._obj_map.items(), headers=table_headers, tablefmt="fancy_grid"
        )
        return f"Registry of {self._name}:\n{table}"

    def __iter__(self) -> Iterator[tuple[str, T]]:
        return iter(self._obj_map.items())

    __str__ = __repr__
