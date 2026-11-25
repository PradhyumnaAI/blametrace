"""Exception-safe PyTorch forward, pre-forward, and backward hook manager."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle


class HookManager:
    """Context-managed PyTorch hook lifecycle guaranteeing zero handle leaks.

    Inside ``session()``, the model is forced into ``eval()`` mode and restored to
    its prior ``training`` flag in the ``finally`` block, even on exception.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._modules: dict[str, nn.Module] = dict(model.named_modules())
        self.activations: dict[str, Any] = {}
        self.pre_activations: dict[str, tuple[Any, ...]] = {}
        self.gradients: dict[str, torch.Tensor] = {}
        self._handles: list[RemovableHandle] = []

    def _resolve_module(self, layer_name: str) -> nn.Module:
        if layer_name not in self._modules:
            raise ValueError(f"Module {layer_name!r} not found in model.")
        return self._modules[layer_name]

    def attach_capture(
        self,
        layer_names: list[str],
        *,
        capture_grads: bool = False,
        capture_pre: bool = False,
        detach_activations: bool = True,
    ) -> None:
        """Register forward, optional pre-forward, and optional backward hooks."""
        for name in layer_names:
            module = self._resolve_module(name)

            if capture_pre:

                def _make_pre(layer: str) -> Callable[..., None]:
                    def _pre_hook(_mod: nn.Module, inputs: tuple[Any, ...]) -> None:
                        if detach_activations:
                            self.pre_activations[layer] = tuple(
                                x.detach() if isinstance(x, torch.Tensor) else x
                                for x in inputs
                            )
                        else:
                            self.pre_activations[layer] = inputs

                    return _pre_hook

                self._handles.append(module.register_forward_pre_hook(_make_pre(name)))

            def _make_fwd(layer: str) -> Callable[..., None]:
                def _fwd_hook(
                    _mod: nn.Module, _inputs: tuple[Any, ...], output: Any
                ) -> None:
                    if detach_activations and isinstance(output, torch.Tensor):
                        self.activations[layer] = output.detach()
                    else:
                        self.activations[layer] = output

                return _fwd_hook

            self._handles.append(module.register_forward_hook(_make_fwd(name)))

            if capture_grads:

                def _make_bwd(layer: str) -> Callable[..., None]:
                    def _bwd_hook(
                        _mod: nn.Module,
                        _grad_in: tuple[torch.Tensor | None, ...],
                        grad_out: tuple[torch.Tensor | None, ...],
                    ) -> None:
                        if grad_out and grad_out[0] is not None:
                            self.gradients[layer] = grad_out[0].detach()

                    return _bwd_hook

                self._handles.append(
                    module.register_full_backward_hook(_make_bwd(name))
                )

    def attach_pre_intervention(
        self,
        layer_name: str,
        transform_fn: Callable[[tuple[Any, ...]], tuple[Any, ...] | None],
    ) -> None:
        """Register a temporary forward pre-hook intervention on module inputs."""
        module = self._resolve_module(layer_name)

        def _pre_intervention_hook(
            _mod: nn.Module, inputs: tuple[Any, ...]
        ) -> tuple[Any, ...] | None:
            return transform_fn(inputs)

        self._handles.append(module.register_forward_pre_hook(_pre_intervention_hook))

    def attach_intervention(
        self,
        layer_name: str,
        transform_fn: Callable[[Any], Any],
    ) -> None:
        """Register a temporary forward output intervention hook for causal ablation."""
        module = self._resolve_module(layer_name)

        def _intervention_hook(
            _mod: nn.Module, _inputs: tuple[Any, ...], output: Any
        ) -> Any:
            return transform_fn(output)

        self._handles.append(module.register_forward_hook(_intervention_hook))

    def clear(self) -> None:
        """Remove all active PyTorch hooks and clear cached tensors."""
        while self._handles:
            handle = self._handles.pop()
            handle.remove()
        self.activations.clear()
        self.pre_activations.clear()
        self.gradients.clear()

    @property
    def active_hook_count(self) -> int:
        """Return number of currently registered PyTorch hook handles."""
        return len(self._handles)

    @contextmanager
    def session(self, *, force_eval: bool = True) -> Iterator[HookManager]:
        """Context manager guaranteeing hook cleanup and eval-mode restoration."""
        was_training = self.model.training
        if force_eval and was_training:
            self.model.eval()
        try:
            yield self
        finally:
            self.clear()
            if force_eval and was_training:
                self.model.train(True)
