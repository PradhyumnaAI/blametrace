"""Task specifications defining target scalar, failure rule, and recovery metric."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class TaskSpec(ABC):
    """Defines how outputs and targets map to failures, scalars, and recovery."""

    name: str = "base"

    @abstractmethod
    def target_scalar(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Return a scalar tensor suitable for backward attribution."""

    @abstractmethod
    def is_failure(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask of shape ``(B,)`` marking failed samples."""

    @abstractmethod
    def recovery_metric(
        self,
        baseline_outputs: torch.Tensor,
        ablated_outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Return scalar recovery metric (higher means stronger causal fix)."""


class ClassificationTask(TaskSpec):
    """Binary (1-logit) or multi-class (C-logit) classification task."""

    name: str = "classification"

    def _predict(self, outputs: torch.Tensor) -> torch.Tensor:
        if outputs.ndim == 1 or outputs.shape[-1] == 1:
            return (outputs.reshape(-1) > 0.0).long()
        return outputs.argmax(dim=-1)

    def target_scalar(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if outputs.ndim == 1 or outputs.shape[-1] == 1:
            logits = outputs.reshape(-1)
            preds = self._predict(outputs)
            # Scalar for predicted (erroneous) class direction
            signed = torch.where(preds == 1, logits, -logits)
            return signed.mean()
        preds = self._predict(outputs)
        chosen = outputs.gather(1, preds.unsqueeze(1)).squeeze(1)
        return chosen.mean()

    def is_failure(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = self._predict(outputs)
        return preds != targets.long().reshape(-1)

    def recovery_metric(
        self,
        baseline_outputs: torch.Tensor,
        ablated_outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Return Flip Rate: fraction of baseline failures fixed by ablation."""
        base_fail = self.is_failure(baseline_outputs, targets)
        n_fail = int(base_fail.sum().item())
        if n_fail == 0:
            return 0.0
        ablated_fail = self.is_failure(ablated_outputs, targets)
        fixed = base_fail & (~ablated_fail)
        return float(fixed.sum().item()) / float(n_fail)


class RegressionTask(TaskSpec):
    """Scalar regression task with configurable absolute-error failure threshold."""

    name: str = "regression"

    def __init__(self, error_threshold: float = 0.5) -> None:
        self.error_threshold = error_threshold

    def target_scalar(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return (outputs.reshape(-1) - targets.reshape(-1)).pow(2).mean()

    def is_failure(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        abs_err = (outputs.reshape(-1) - targets.reshape(-1)).abs()
        return abs_err > self.error_threshold

    def recovery_metric(
        self,
        baseline_outputs: torch.Tensor,
        ablated_outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Return mean absolute error reduction (positive means improved)."""
        base_err = (baseline_outputs.reshape(-1) - targets.reshape(-1)).abs().mean()
        abl_err = (ablated_outputs.reshape(-1) - targets.reshape(-1)).abs().mean()
        return float((base_err - abl_err).item())


class SegmentationTask(TaskSpec):
    """Binary dense segmentation task evaluated with Dice recovery."""

    name: str = "segmentation"

    def __init__(self, dice_threshold: float = 0.7, eps: float = 1e-6) -> None:
        self.dice_threshold = dice_threshold
        self.eps = eps

    def _sample_dice(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        preds = (outputs > 0.0).float().reshape(outputs.shape[0], -1)
        gt = (targets > 0.5).float().reshape(targets.shape[0], -1)
        inter = (preds * gt).sum(dim=1)
        denom = preds.sum(dim=1) + gt.sum(dim=1)
        return (2.0 * inter + self.eps) / (denom + self.eps)

    def target_scalar(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        preds = outputs > 0.0
        gt = targets > 0.5
        err_mask = preds ^ gt
        if bool(err_mask.any().item()):
            return outputs[err_mask].abs().mean()
        return outputs.abs().mean()

    def is_failure(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._sample_dice(outputs, targets) < self.dice_threshold

    def recovery_metric(
        self,
        baseline_outputs: torch.Tensor,
        ablated_outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Return mean Dice gain achieved by ablation."""
        base_dice = self._sample_dice(baseline_outputs, targets).mean()
        abl_dice = self._sample_dice(ablated_outputs, targets).mean()
        return float((abl_dice - base_dice).item())


def resolve_task(task: TaskSpec | str | None) -> TaskSpec:
    """Resolve a task name or instance into a concrete TaskSpec."""
    if isinstance(task, TaskSpec):
        return task
    if task is None or task == "classification":
        return ClassificationTask()
    if task == "regression":
        return RegressionTask()
    if task == "segmentation":
        return SegmentationTask()
    raise ValueError(
        f"Unknown task {task!r}. Expected 'classification', 'regression', or 'segmentation'."
    )
