---
name: New Architecture Adapter
about: Propose a new PyTorch model family adapter
labels: enhancement, adapter
---

**Architecture Family:** (e.g., Graph Neural Network, Diffusion UNet, RWKV)
**Raw Layer Output Structure:** (e.g., `(B, Nodes, D)` tensor or tuple)
**Proposed Canonical `(B, U, P)` Mapping:**
- `U` (Units to blame):
- `P` (Positions):
**Proposed Planted-Shortcut Benchmark:** (How will CI prove this adapter catches a ground-truth failure?)
