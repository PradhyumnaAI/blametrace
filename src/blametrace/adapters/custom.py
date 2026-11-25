"""Universal fallback adapter for any custom PyTorch nn.Module."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from blametrace.adapters.base import BaseAdapter


class CustomAdapter(BaseAdapter):
    """Handles arbitrary PyTorch modules returning Tensors, Tuples, Lists, or Dicts."""

    name: str = "custom"
    priority: int = -100

    def __init__(
        self,
        target_layers: list[str] | None = None,
        unit_dim: int = 1,
        primary_key: str | int = 0,
    ) -> None:
        self.target_layers = target_layers
        self.unit_dim = unit_dim
        self.primary_key = primary_key

    @classmethod
    def matches(cls, model: nn.Module) -> bool:
        return True

    def discover_targets(self, model: nn.Module) -> list[str]:
        if self.target_layers is not None:
            named = dict(model.named_modules())
            missing = [name for name in self.target_layers if name not in named]
            if missing:
                raise ValueError(f"Target layers not found in model: {missing}")
            return list(self.target_layers)

        leaf_modules = [
            name
            for name, mod in model.named_modules()
            if name and len(list(mod.children())) == 0
        ]
        if not leaf_modules:
            raise ValueError("Model has no hookable child modules.")
        return [leaf_modules[-2] if len(leaf_modules) >= 2 else leaf_modules[-1]]

    def _extract_primary(self, raw_output: Any) -> torch.Tensor:
        if isinstance(raw_output, torch.Tensor):
            return raw_output
        if isinstance(raw_output, (tuple, list)):
            idx = int(self.primary_key)
            val = raw_output[idx]
            if not isinstance(val, torch.Tensor):
                raise TypeError(f"Element {idx} of output sequence is not a Tensor.")
            return val
        if isinstance(raw_output, dict):
            val = raw_output[self.primary_key]
            if not isinstance(val, torch.Tensor):
                raise TypeError(
                    f"Key {self.primary_key!r} of output dict is not a Tensor."
                )
            return val
        raise TypeError(f"Unsupported module output type: {type(raw_output)!r}")

    def to_canonical(self, raw_output: Any) -> torch.Tensor:
        tensor = self._extract_primary(raw_output)
        if tensor.ndim < 2:
            raise ValueError(
                f"Expected tensor with at least 2 dims (B, U, ...), got {tensor.shape}."
            )
        if tensor.ndim == 2:
            return tensor.unsqueeze(-1)

        unit_axis = self.unit_dim if self.unit_dim >= 0 else tensor.ndim + self.unit_dim
        if unit_axis <= 0 or unit_axis >= tensor.ndim:
            raise ValueError(
                f"Invalid unit_dim={self.unit_dim} for tensor of shape {tensor.shape}."
            )

        moved = tensor.movedim(unit_axis, 1)
        batch_size, num_units = moved.shape[0], moved.shape[1]
        return moved.reshape(batch_size, num_units, -1)

    def from_canonical(self, canonical: torch.Tensor, raw_template: Any) -> Any:
        template_tensor = self._extract_primary(raw_template)
        if template_tensor.ndim == 2:
            restored = canonical.squeeze(-1)
        else:
            unit_axis = (
                self.unit_dim
                if self.unit_dim >= 0
                else template_tensor.ndim + self.unit_dim
            )
            moved_shape = list(template_tensor.movedim(unit_axis, 1).shape)
            restored = canonical.reshape(*moved_shape).movedim(1, unit_axis)

        if isinstance(raw_template, torch.Tensor):
            return restored
        if isinstance(raw_template, tuple):
            idx = int(self.primary_key)
            items = list(raw_template)
            items[idx] = restored
            return tuple(items)
        if isinstance(raw_template, list):
            idx = int(self.primary_key)
            items = list(raw_template)
            items[idx] = restored
            return items
        if isinstance(raw_template, dict):
            updated = dict(raw_template)
            updated[self.primary_key] = restored
            return updated
        raise TypeError(f"Unsupported template type: {type(raw_template)!r}")
