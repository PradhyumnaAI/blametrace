"""Architecture-agnostic core: hooks, triage, causal proof, registry, engine.

Invariant 2: nothing in this package may branch on a model family. All
family-specific handling lives in ``blametrace.adapters``.
"""

from blametrace.core.causal import clean_mean_ablate
from blametrace.core.hooks import HookManager
from blametrace.core.registry import (
    auto_detect_adapter,
    get_adapter,
    list_adapters,
    register_adapter,
    unregister_adapter,
)
from blametrace.core.tasks import (
    ClassificationTask,
    RegressionTask,
    SegmentationTask,
    TaskSpec,
    resolve_task,
)

__all__ = [
    "ClassificationTask",
    "HookManager",
    "RegressionTask",
    "SegmentationTask",
    "TaskSpec",
    "auto_detect_adapter",
    "clean_mean_ablate",
    "get_adapter",
    "list_adapters",
    "register_adapter",
    "resolve_task",
    "unregister_adapter",
]
