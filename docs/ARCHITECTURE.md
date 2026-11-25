# Architecture

This document describes how BlameTrace is designed. The scope and rules are in
[`FREEZE.md`](../FREEZE.md). Nothing here is implemented yet; each part is built
by its milestone in [`ROADMAP.md`](ROADMAP.md).

## Goal

Given a trained PyTorch model and its failure cases (misclassified samples,
high-error regressions, or high-error segmentation masks), BlameTrace:

1. Narrows hundreds of internal units to a short list of suspects.
2. Proves which suspects caused the failures by clean-mean ablation, against a
   random-unit null.
3. Reports the result offline, with an optional LLM narrative that only
   describes the evidence.

Target latency is under 30 seconds on a single GPU. This is a goal, not a
measured result, and will be published per adapter only after measurement.

## Runtime flow

```
bt.audit(model, failures, clean, task=None, metadata=None, explainer=None)
   |
   v
Auditor (core/engine.py)
   |
   |  1. Registry picks an adapter           core/registry.py
   |  2. Task defines failure + metric       core/tasks.py
   v
Stage 1  Triage                              core/triage.py
   Grad x Act and failure-vs-clean activation difference
   -> Top-K suspect units (K <= 12)
   |
   v
Stage 2  Causal proof                        core/causal.py
   Clean-mean ablation on suspects only
   Null control (random units, same layer)
   Collateral check (ablate on clean samples)
   Cosine clustering (> 0.8) -> FailurePattern
   Joint-ablation score per pattern
   |
   v
Stage 3  Translation                         report/, explainers/
   Terminal summary + HTML + JSON            always, offline
   Optional LLM narrative                    opt-in, evidence only
   |
   v
(optional) Slice attribution                 core/slices.py
   Which metadata groups trigger each pattern
```

## Modules

| Module | Responsibility | Must not |
|---|---|---|
| `types.py` | Pydantic result models, `schema_version` | hold tensors |
| `core/hooks.py` | `HookManager.session()`, handle cleanup, train-mode restore | leave handles behind |
| `core/tasks.py` | target scalar, failure rule, recovery metric | know about model families |
| `core/triage.py` | failure collection, attribution, Top-K | claim causality |
| `core/causal.py` | ablation, null control, clustering | branch on architecture |
| `core/slices.py` | metadata association tests with FDR correction | label associations as causes |
| `core/registry.py` | `@register_adapter`, priority-ordered `matches()` | contain `isinstance` logic |
| `core/engine.py` | `Auditor` orchestration | duplicate stage logic |
| `adapters/*` | the `(B, U, P)` contract per model family | leak family logic into `core/` |
| `explainers/*` | narrate evidence | pick culprits or reach the network by default |
| `report/*` | render results | make network calls |

## The `(B, U, P)` contract

Every adapter exposes:

- `to_canonical(raw) -> Tensor` of shape `(B, U, P)`.
- `from_canonical(canonical, raw_template) -> raw`, restoring the original
  Tensor, tuple, list, dict or `ModelOutput`.
- `matches(model) -> int | None`, a priority used by the registry.

Round-trip rule: `from_canonical(to_canonical(x), x)` equals `x` exactly for
every non-ablated unit. See [ADR-0001](adr/0001-canonical-tensor-contract.md).

| Adapter | Target models | Raw output | Canonical `(B, U, P)` | Diagnostic focus |
|---|---|---|---|---|
| `ANNAdapter` | MLP, tabular | `(B, D)` | `(B, D, 1)` | shortcut input columns, culprit neurons, dead units |
| `CNNAdapter` | torchvision, timm CNNs | `(B, C, H, W)` | `(B, C, H*W)` | texture, glare, watermark shortcuts, channel clustering |
| `RNNAdapter` | RNN, LSTM, GRU | `(out, state)` tuple | `(B, H, T)` | failure timesteps, trigger prefixes, safe state repacking |
| `TransformerAdapter` | NLP and ViT transformers | `(B, L, D)` or `(B, H, L, d)` | `(B, Heads, L)` | culprit attention heads, patch and token shortcuts |
| `CNNMambaAdapter` | hybrid CNN-Mamba | dual streams plus gates | parallel `(B, C, H*W)` and `(B, D, L)` | local vs global blame split, fusion-gate health |
| `CustomAdapter` | any `nn.Module` | Tensor, tuple, dict | auto-unwrapped primary tensor | universal fallback |
| `CompositeAdapter` | hybrid models | per submodule | per submodule | routes each submodule to an adapter |

The clean-mean baseline defaults to shape `(1, U, 1)`; `(1, U, P)` is optional.
Attention heads carry a `head_dim` slice per position. The adapter ablates the
whole slice and reduces to a scalar per head only for attribution.

## Composite models

Hybrid models route per submodule. `CompositeAdapter` discovers submodules (for
example a ResNet backbone and an LSTM decoder), picks an adapter for each
through the registry, and prefixes unit IDs with the module path, such as
`backbone.layer4/ch17`.

## Known hazards the adapters must handle

- **cuDNN RNN backward** fails in eval mode on GPU. Triage disables cuDNN for
  that pass.
- **Packed sequences** are unpacked and repacked; padding is excluded from the
  clean-mean baseline.
- **Fused attention** (SDPA, FlashAttention) never materializes attention
  weights, so heads are sliced at the output projection input.
- **GQA and MQA** have fewer key/value heads than query heads; units are query
  heads.
- **Inplace activations** are detected and cloned so gradients stay correct.
- **Functional activations** have no module to hook; use an explicit layer list.

## Causal proof

For each suspect unit or cluster:

1. Ablate on the failure subset. Record `flip_rate`, `confidence_drop`, and
   task-specific recovery (Dice or IoU for segmentation).
2. Ablate an equal number of random units from the same layer, many draws.
   Report the effect as a z-score against that null.
3. Ablate the same suspects on clean samples. Report collateral damage.
4. Keep only suspects that beat the null and spare clean samples.

See [ADR-0002](adr/0002-causal-proof-requires-null-control.md). "Causal" means
the interventional effect on model outputs. It does not mean root cause in the
data.

## Slice attribution

Optional, offline, one extra forward pass. Compute each pattern's activation on
every sample, test association with metadata columns (Fisher or chi-square, FDR
correction, minimum group size), and report correlated columns together.
Results use the `SliceAssociation` type and are always phrased as "associated
with". The interventional line, "ablating the pattern fixes X% of group G", is
measured.

## Report layers

| Layer | Network | Always available |
|---|---|---|
| Terminal summary | none | yes |
| Single-file HTML | none | yes |
| JSON (`schema_version`) | none | yes |
| LLM narrative | opt-in | no |

LLM text is HTML-escaped, labeled as a hypothesis, and validated against the
evidence it cites. See [ADR-0003](adr/0003-llm-narrates-never-selects.md).
