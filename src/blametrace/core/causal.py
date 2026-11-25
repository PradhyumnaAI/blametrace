"""Architecture-agnostic clean-mean causal ablation on canonical (B, U, P) tensors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from blametrace.adapters.base import BaseAdapter
from blametrace.core.hooks import HookManager


def clean_mean_ablate(
    model: nn.Module,
    adapter: BaseAdapter,
    layer_name: str,
    unit_indices: list[int],
    clean_unit_means: torch.Tensor,
    failure_inputs: Any,
    forward_fn: Callable[[nn.Module, Any], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Run ``model`` on ``failure_inputs`` replacing ``unit_indices`` with clean means.

    Args:
        model: Trained PyTorch module.
        adapter: Adapter translating raw outputs to/from canonical ``(B, U, P)``.
        layer_name: Target module path to intervene on.
        unit_indices: Indices along the ``Units`` axis to replace with clean means.
        clean_unit_means: Tensor of shape ``(U,)``, ``(1, U, 1)``, or ``(1, U, P)``.
        failure_inputs: Input batch fed to ``model``.
        forward_fn: Optional custom call wrapper ``(model, inputs) -> outputs``.

    Returns:
        Model outputs under causal intervention.
    """
    if clean_unit_means.ndim == 1:
        mean_view = clean_unit_means.view(1, -1, 1)
    elif clean_unit_means.ndim == 3:
        mean_view = clean_unit_means
    else:
        raise ValueError(
            "clean_unit_means must have shape (U,), (1, U, 1), or (1, U, P), "
            f"got {clean_unit_means.shape}."
        )

    def _ablate_output(raw_output: Any) -> Any:
        canonical = adapter.to_canonical(raw_output).clone()
        target_means = mean_view.to(device=canonical.device, dtype=canonical.dtype)
        canonical[:, unit_indices, :] = target_means[:, unit_indices, :]
        return adapter.from_canonical(canonical, raw_output)

    def _default_forward(m: nn.Module, x: Any) -> torch.Tensor:
        out = m(x)
        if not isinstance(out, torch.Tensor):
            raise TypeError(
                f"Expected model forward to return a Tensor, got {type(out)!r}. "
                "Pass a custom forward_fn to extract logits."
            )
        return out

    runner = forward_fn if forward_fn is not None else _default_forward
    manager = HookManager(model)
    with manager.session():
        manager.attach_intervention(layer_name, _ablate_output)
        with torch.no_grad():
            return runner(model, failure_inputs)
