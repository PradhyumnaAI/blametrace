"""Shared unit-test fixtures enforcing Invariant 5 (zero leftover PyTorch hooks)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch.nn as nn


def _count_model_hooks(model: nn.Module) -> int:
    total = 0
    for mod in model.modules():
        total += len(mod._forward_hooks)
        total += len(mod._forward_pre_hooks)
        total += len(mod._backward_hooks)
        total += len(mod._backward_pre_hooks)
    return total


@pytest.fixture(autouse=True)
def assert_zero_leftover_hooks(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    """Autouse fixture verifying any tracked nn.Module has zero hooks after a test."""
    tracked: list[nn.Module] = []
    request.node.tracked_models = tracked
    yield
    for model in tracked:
        leftover = _count_model_hooks(model)
        assert leftover == 0, (
            f"Detected {leftover} leftover hook(s) on {type(model).__name__}."
        )
