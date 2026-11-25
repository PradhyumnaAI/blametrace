"""Architecture-agnostic core: hooks, triage, causal proof, registry, engine.

Invariant 2: nothing in this package may branch on a model family. All
family-specific handling lives in ``blametrace.adapters``.
"""
