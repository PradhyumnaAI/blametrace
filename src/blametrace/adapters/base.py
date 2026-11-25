"""Abstract BaseAdapter enforcing the canonical (Batch, Units, Positions) contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseAdapter(ABC):
    """Base class for all architecture-specific BlameTrace adapters."""

    name: str = "base"
    priority: int = 0

    @classmethod
    def matches(cls, model: nn.Module) -> bool:
        """Return True if this adapter automatically matches ``model``."""
        return False

    @abstractmethod
    def discover_targets(self, model: nn.Module) -> list[str]:
        """Return qualified module names to hook for attribution and ablation."""

    @abstractmethod
    def to_canonical(self, raw_output: Any) -> torch.Tensor:
        """Convert raw module output into a 3D tensor of shape ``(B, U, P)``."""

    @abstractmethod
    def from_canonical(self, canonical: torch.Tensor, raw_template: Any) -> Any:
        """Restore modified canonical tensor back to the exact raw output structure."""

    def render_evidence(
        self,
        inputs: Any,
        canonical_activations: torch.Tensor,
        unit_indices: list[int],
    ) -> dict[str, Any]:
        """Produce structured offline evidence for verified culprit units."""
        selected = canonical_activations[:, unit_indices, :]
        mean_per_unit: list[float] = [
            float(v) for v in selected.mean(dim=(0, 2)).detach().cpu().tolist()
        ]
        return {
            "unit_indices": list(unit_indices),
            "mean_failure_activation": mean_per_unit,
        }
