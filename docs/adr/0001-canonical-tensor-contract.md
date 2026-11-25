# ADR-0001: Canonical `(Batch, Units, Positions)` tensor contract

## Status
Accepted

## Context
PyTorch architectures emit incompatible intermediate representations:

- Dense MLPs emit `(B, D)`.
- 2D CNNs emit `(B, C, H, W)`.
- Recurrent layers emit nested tuples `(out (B, T, H), (h_n, c_n))`.
- Transformers emit `(B, L, D)` or `(B, Heads, L, d_head)`.
- Hybrid CNN-Mamba models emit parallel spatial and sequence streams plus gate
  tensors.

Writing architecture-specific triage and ablation loops inside `core/` would
create fragile `isinstance` branching and duplicate causal math across every
model family.

## Decision
All adapters implement:

- `to_canonical(raw_output) -> torch.Tensor` returning shape `(Batch, Units,
  Positions)`.
- `from_canonical(canonical, raw_template) -> Any` restoring the exact upstream
  PyTorch object, including nested tuples, lists, dicts or `ModelOutput`s.

Rules that make the contract lossless:

- Non-ablated units round-trip bit-identical.
- The clean-mean baseline has shape `(1, U, 1)` by default, `(1, U, P)`
  optionally.
- Units that carry extra structure, such as an attention head's `head_dim`
  slice, are ablated whole. They are reduced to one scalar per position only
  for attribution.

## Consequences
- `core/triage.py` and `core/causal.py` are written once and verified once.
- Adding a model family means writing a thin `BaseAdapter` subclass without
  touching core attribution or ablation code.
- Each adapter ships a round-trip identity test.
