"""Adapters map any model family onto the canonical (B, U, P) tensor contract.

Invariant 1: every adapter implements ``to_canonical`` and ``from_canonical``
and restores the exact original object during ablation.
"""

from blametrace.adapters.base import BaseAdapter
from blametrace.adapters.custom import CustomAdapter

__all__ = ["BaseAdapter", "CustomAdapter"]
