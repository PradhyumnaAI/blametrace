# Execution roadmap

Task checklist per milestone. Exact gate conditions and scope rules live in
[`FREEZE.md`](../FREEZE.md), which wins if the two ever disagree.

Milestone N+1 does not begin until Milestone N's gate passes in CI.

## Milestone 0: Foundation (scaffold)

- [ ] `types.py`: `CulpritUnit`, `BranchBlameSplit`, `FailurePattern`, `AuditReport` with `schema_version`
- [ ] `core/hooks.py`: `HookManager.session()` with forward, pre and backward hooks, eval mode forced and restored
- [ ] `core/tasks.py`: target scalar, failure rule, recovery metric
- [ ] `core/causal.py`: `clean_mean_ablate` on canonical `(B, U, P)`
- [ ] `adapters/base.py` and `adapters/custom.py`
- [ ] `core/registry.py`: `@register_adapter`, priority-ordered `matches()`
- [ ] Autouse pytest fixture asserting zero leftover hooks
- [ ] Contract spike: LSTM tuple, attention head, Mamba-style boundary round-trip
- [ ] CI green on Python 3.11, 3.12, 3.13

**Gate:** deterministic linear-graph test passes exact causal math, zero hook
leaks on forced crash, contract spike round-trips.

## Milestone 1: ANN / MLP (v0.1.0)

- [ ] `adapters/ann.py`: `nn.Linear` discovery, `(B, D) -> (B, D, 1)`, dead-unit checks
- [ ] `core/triage.py`: failure collector, Grad x Act, activation difference, optional Integrated Gradients
- [ ] `core/causal.py`: null control, collateral check, cosine clustering, joint ablation
- [ ] `core/engine.py`: `Auditor.audit()` end to end
- [ ] `report/terminal.py` and `report/html.py`: offline single-file report
- [ ] `blametrace.audit(...)` one-line entry point
- [ ] `tests/benchmarks/test_m1_ann_shortcut.py`

**Gate:** planted leaked column isolated, failures recovered, random ablation
recovers far less, negative control clean, 5+ seeds, under 5 s on CPU.

## Milestone 1.x: Slices and explainer (v0.1.x)

- [ ] `core/slices.py`: association tests with FDR correction, `SliceAssociation` type
- [ ] `explainers/base.py` and `explainers/openai_compat.py`
- [ ] Offline-invariant test (network calls blocked)

**Gate:** planted leak limited to one metadata group is reported with that group.

## Milestone 2: CNN (v0.2.0)

- [ ] `adapters/cnn.py`: late `Conv2d` discovery in `torchvision` and `timm`
- [ ] Spatial cosine channel clustering above 0.8
- [ ] `core/visuals.py`: bilinear upsampling, paired dimmed-context and crop sheets
- [ ] Base64 WebP report cards under 3 MB

**Gate:** planted corner watermark or glare shortcut.

## Milestone 3: RNN / LSTM / GRU (v0.3.0)

- [ ] `adapters/rnn.py`: safe `(out, (h_n, c_n))` unpacking and repacking
- [ ] Timestep attribution along `P = T`, bidirectional support
- [ ] Packed-sequence handling, cuDNN-safe backward
- [ ] Highlighted span evidence renderer

**Gate:** planted boilerplate prefix trigger.

## Milestone 4: Transformers (v0.4.0)

- [ ] `adapters/transformer.py`: attention-head slicing at the output projection
- [ ] GQA and MQA, fused attention (SDPA, FlashAttention)
- [ ] Token-span (NLP) and patch-grid (ViT) evidence renderer
- [ ] `adapters/composite.py`: submodule routing for hybrids

**Gate:** planted attention-sink or corner-patch shortcut.

## v1.0.0

- [ ] API stabilization, docs complete, all gates green

## Milestone 5: Hybrid CNN-Mamba and segmentation (v0.5.0, experimental)

- [ ] `adapters/cnn_mamba.py`: boundary-only hooks preserving CUDA `selective_scan` speed
- [ ] Two-tier ablation: branch blame split, then top channels
- [ ] Fusion-gate collapse telemetry
- [ ] Segmentation FP/FN pixel error masks with Dice/IoU recovery

**Gate:** planted border artifact in a lightweight CNN-Mamba segmenter.
