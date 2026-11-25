"""BlameTrace: causal failure debugging for PyTorch models.

Pre-alpha. Milestone 0 foundation is available; ``blametrace.audit`` arrives in M1.
"""

from blametrace.adapters import BaseAdapter, CustomAdapter
from blametrace.core import (
    ClassificationTask,
    HookManager,
    RegressionTask,
    SegmentationTask,
    TaskSpec,
    auto_detect_adapter,
    clean_mean_ablate,
    get_adapter,
    list_adapters,
    register_adapter,
    resolve_task,
)
from blametrace.types import (
    SCHEMA_VERSION,
    AuditReport,
    BranchBlameSplit,
    CulpritUnit,
    FailurePattern,
)

__version__ = "0.0.1"

__all__ = [
    "SCHEMA_VERSION",
    "AuditReport",
    "BaseAdapter",
    "BranchBlameSplit",
    "ClassificationTask",
    "CulpritUnit",
    "CustomAdapter",
    "FailurePattern",
    "HookManager",
    "RegressionTask",
    "SegmentationTask",
    "TaskSpec",
    "__version__",
    "auto_detect_adapter",
    "clean_mean_ablate",
    "get_adapter",
    "list_adapters",
    "register_adapter",
    "resolve_task",
]
