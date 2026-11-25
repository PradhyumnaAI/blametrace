# BlameTrace project freeze

**Status:** FROZEN as of 2026-11-25. Version of this document: 1.0.

This file is the scope contract for BlameTrace. Anything not listed under
"In scope" is out of scope until the current milestone's gate passes and an RFC
is accepted. If a good idea arrives mid-milestone, it goes into
[`docs/rfcs/`](docs/rfcs/0000-template.md) and waits.

## 1. Mission

Given a trained PyTorch model and its failure cases, find the internal units
that caused those failures and prove it by intervention.

The strongest claim we make, and the one the benchmarks back:

> BlameTrace finds and proves shortcut and spurious-feature dependence behind
> a group of model failures.

We do **not** claim to explain every failure, to find root causes in the data,
or to be a general interpretability dashboard.

## 2. Five frozen invariants

1. **Canonical `(B, U, P)` contract.** Every adapter maps raw module output to
   `(Batch, Units, Positions)` via `to_canonical` and restores the exact
   original object via `from_canonical`. Non-ablated units must round-trip
   bit-identical.
2. **Zero architecture branching in `core/`.** No `isinstance(model, ...)` or
   model-specific logic in `src/blametrace/core/`. Family logic lives only in
   `src/blametrace/adapters/`.
3. **Causal proof mandate.** Every reported culprit carries a measured
   clean-mean ablation score (`confidence_drop` and `flip_rate`), compared
   against a random-unit null.
4. **Offline-first.** Attribution, ablation, terminal summary and HTML report
   work with zero network calls when `explainer=None`.
5. **Zero-leak hook hygiene.** Every hook runs inside
   `HookManager.session()` and is removed in a `finally:` block.

## 3. Accepted amendments (part of the freeze)

| # | Amendment |
|---|---|
| A1 | Null control and collateral-damage check in every causal proof |
| A2 | Joint-ablation score per `FailurePattern`, not only per unit |
| A3 | Triage ranks by Grad x Act and failure-vs-clean activation difference, union capped at K <= 12; Integrated Gradients optional |
| A4 | `core/tasks.py` defines target scalar, failure rule, and recovery metric per task |
| A5 | Contract details: baseline shape `(1, U, 1)` by default, round-trip identity tests, pre-hook support, eval mode forced and restored, inplace-activation detection, explicit layer list for functional activations |
| A6 | Milestone 1 gate rewritten (see section 6) |
| A7 | Explainer hardening: escaped HTML, evidence-validated claims, opt-in cloud, sync wrapper, `httpx` behind `blametrace[llm]` |
| A8 | `adapters/composite.py` routes submodules to adapters for hybrid models |
| A9 | Data slice attribution (`core/slices.py`) as v0.1.x, outside the M1 gate |

Support matrix: eval mode, single device, autocast allowed with fp32 baseline
accumulation. `torch.compile`, FSDP and DDP are unsupported.

## 4. In scope

- Adapters: ANN, CNN, RNN, Transformer, CNN-Mamba, Custom, Composite.
- Tasks: classification, regression, segmentation.
- Stage 1 triage, Stage 2 causal proof, Stage 3 offline report.
- Optional OpenAI-compatible explainer (Ollama, vLLM, OpenAI, Gemini).
- Data slice attribution from user-supplied metadata.
- Terminal summary, single-file HTML report, versioned JSON report.
- One-line entry point: `blametrace.audit(model, failures, clean)`.

## 5. Out of scope until after v1.0

- Mixture-of-Experts and graph-neural-network dedicated adapters.
- Generative and diffusion model support beyond user-defined `Task`s.
- `torch.compile`, FSDP, DDP support.
- Automatic data cleaning, label-error detection, root-cause inference.
- Interactive dashboard or hosted service.
- Fairness auditing.
- Any change to the five invariants.

## 6. Milestones and release gates

Build strictly in order. Milestone N+1 does not begin until Milestone N's
benchmark passes in CI.

| Milestone | Release | Scope | Gate |
|---|---|---|---|
| M0 | scaffold | `types.py`, `HookManager`, `tasks.py`, `clean_mean_ablate`, `BaseAdapter`, `CustomAdapter`, registry | Unit tests green, leak fixture active, contract spike (LSTM tuple, attention head, Mamba boundary) round-trips |
| M1 | v0.1.0 | `ANNAdapter`, triage, causal proof, `Auditor`, offline HTML report, `bt.audit` | See M1 gate below |
| M1.x | v0.1.x | `core/slices.py`, `explainers/openai_compat.py` | Planted group-limited leak is reported with its group |
| M2 | v0.2.0 | `CNNAdapter`, `core/visuals.py`, visual evidence cards | Planted corner watermark or glare benchmark |
| M3 | v0.3.0 | `RNNAdapter`, packed sequences, cuDNN-safe backward | Planted boilerplate prefix benchmark |
| M4 | v0.4.0 | `TransformerAdapter`, head slicing, GQA/MQA, fused attention | Planted attention-sink or corner-patch benchmark |
| v1.0 | v1.0.0 | Stabilize API after M4 | All gates green, docs complete |
| M5 | v0.5.0 experimental | `CNNMambaAdapter`, two-tier ablation, segmentation Dice/IoU | Planted border-artifact benchmark (experimental label) |

### M1 gate (all conditions must hold)

1. The top input-attribution feature is the planted column.
2. Ablating the culprit cluster recovers at least 60% of failure accuracy.
3. Ablating the same number of random units recovers at most 20%.
4. A negative-control model trained without the shortcut yields no
   statistically significant culprits.
5. Passes deterministically across 5 or more seeds.
6. Runs in under 5 seconds on CPU.

Thresholds in items 2 and 3 may be tightened but never loosened without an RFC.

## 7. Frozen repository structure

```
src/blametrace/
  types.py            Pydantic models, schema_version
  core/               hooks, triage, causal, tasks, slices, registry, engine, visuals
  adapters/           base, custom, composite, ann, cnn, rnn, transformer, cnn_mamba
  explainers/         base, openai_compat
  report/             terminal, html
tests/unit/           fast CPU tests, under 10 seconds total
tests/benchmarks/     planted-shortcut gates
docs/                 ARCHITECTURE, ROADMAP, adr/, rfcs/
.github/              CI, issue and pull request templates, dependabot
```

New code goes only in these locations. Modules listed above are created by
their milestone, not before.

Packaging: core dependencies are `torch` and `pydantic` only. Optional extras:
`llm` (`httpx`) and `vision` (`pillow`, `torchvision`).

## 8. Engineering rules

- Python 3.11+, `src/` layout, Hatchling, `uv`.
- `ruff` lint and format: 88 columns, double quotes, sorted imports, `X | None`.
- `mypy --strict` clean.
- No placeholders or TODOs in merged code.
- Every change ships with tests: `(B, U, P)` shape parity, zero hook leaks,
  causal recovery where relevant.
- An autouse pytest fixture asserts all forward, forward-pre and backward hook
  dicts are empty after every test.
- Latency numbers appear in docs only after being measured per adapter.
- `AuditReport` is JSON-serializable with a `schema_version`; no tensors in
  Pydantic fields.
- The LLM explains evidence and never selects culprits.

## 9. Change control

To change anything in this document:

1. Open an RFC from `docs/rfcs/0000-template.md`.
2. State which section changes, why, and what it costs.
3. Changes to sections 2, 6 or 7 need the current milestone's gate passing
   first.
4. On acceptance, update this file, bump its version, and record the date.

Bug fixes, tests, docs, and performance work that do not change scope need no
RFC.

## 10. Current status

| Item | State |
|---|---|
| Repository, license, docs, governance, CI, packaging | Initialized |
| Package skeleton | Docstring-only modules, no implementation |
| Milestone 0 | Next: first implementation task |
| Milestone 1 | Blocked on M0 |

Open items before the first public release:

- Register `blametrace` as a PyPI Pending Publisher and claim the name on the
  first release (`v0.0.1` or `v0.1.0`).
- Confirm the pinned GitHub Action versions in `.github/workflows/ci.yml` on the
  first CI run.
