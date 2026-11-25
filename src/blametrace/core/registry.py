"""Adapter registry with priority-ordered automatic architecture matching."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from blametrace.adapters.base import BaseAdapter
from blametrace.adapters.custom import CustomAdapter

_ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {
    "custom": CustomAdapter,
}


def register_adapter(
    name: str, *, priority: int | None = None
) -> Callable[[type[BaseAdapter]], type[BaseAdapter]]:
    """Class decorator registering a BaseAdapter subclass under ``name``."""

    def _decorator(cls: type[BaseAdapter]) -> type[BaseAdapter]:
        if not issubclass(cls, BaseAdapter):
            raise TypeError(f"{cls.__name__} must inherit from BaseAdapter.")
        key = name.strip().lower()
        cls.name = key
        if priority is not None:
            cls.priority = priority
        _ADAPTER_REGISTRY[key] = cls
        return cls

    return _decorator


def unregister_adapter(name: str) -> None:
    """Remove a registered adapter by name (primarily for test isolation)."""
    key = name.strip().lower()
    if key == "custom":
        raise ValueError("Cannot unregister the built-in 'custom' fallback adapter.")
    _ADAPTER_REGISTRY.pop(key, None)


def get_adapter(name: str, **kwargs: Any) -> BaseAdapter:
    """Instantiate a registered adapter by name."""
    key = name.strip().lower()
    if key not in _ADAPTER_REGISTRY:
        available = sorted(_ADAPTER_REGISTRY.keys())
        raise KeyError(f"Unknown adapter {name!r}. Registered adapters: {available}")
    return _ADAPTER_REGISTRY[key](**kwargs)


def list_adapters() -> list[str]:
    """Return adapter names sorted by descending priority, then name."""
    ordered = sorted(
        _ADAPTER_REGISTRY.items(),
        key=lambda item: (-item[1].priority, item[0]),
    )
    return [name for name, _ in ordered]


def auto_detect_adapter(model: nn.Module, **kwargs: Any) -> BaseAdapter:
    """Select the highest-priority registered adapter whose ``matches(model)`` is True."""
    ordered = sorted(
        _ADAPTER_REGISTRY.values(),
        key=lambda cls: (-cls.priority, cls.name),
    )
    for adapter_cls in ordered:
        if adapter_cls.matches(model):
            return adapter_cls(**kwargs)
    return CustomAdapter(**kwargs)
