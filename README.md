# BlameTrace

**Find and prove the internal shortcuts behind your PyTorch model's failures.**

BlameTrace takes a trained model and the cases it got wrong, then tells you
which internal units (CNN channels, neurons, attention heads) caused those
failures. It proves each claim by intervention: swap the suspect units for
their clean-data average and measure how many failures disappear.

> **Status: pre-alpha, design frozen.** This repository currently holds the
> design, governance and CI. No implementation is published yet, and the API
> below is the planned one. See [`FREEZE.md`](FREEZE.md) and
> [`docs/ROADMAP.md`](docs/ROADMAP.md).

## The problem

When a model fails, today you can see that it failed and little else.

- Saliency maps and attribution scores say a region or neuron was *important*.
  Important is not the same as *responsible*, and these tools rarely test it.
- Aggregate metrics say accuracy dropped on a slice, not why.

So you guess: a leaked column, a watermark, a boilerplate prefix, a border
artifact. Finding out can take days of manual probing.

BlameTrace answers one question:

> Which internal parts of this model caused these specific failures, and can we
> prove it by intervention?

## Planned usage

```python
import blametrace as bt

report = bt.audit(model, failures=wrong_samples, clean=good_samples)

report.show()  # terminal summary
report.to_html("audit.html")  # single-file, shareable, works offline
report.to_json("audit.json")  # versioned schema, for CI
```

Optional metadata lets the report say which groups of your data trigger a
pattern:

```python
report = bt.audit(
    model,
    failures,
    clean,
    metadata=df[["supplier", "camera", "month"]],
)
```

For models the built-in adapters do not cover, a `CustomAdapter` accepts an
explicit list of layers to inspect.

## What the report will contain

1. **Ranked culprit units**, each with a measured `confidence_drop` and
   `flip_rate`, its comparison against a random-ablation null, and collateral
   damage on clean samples.
2. **Failure patterns**: clusters of correlated units with a joint-ablation
   score. Hundreds of units become a handful of findings.
3. **Recovery numbers**, for example "ablating pattern A fixes 78% of failures
   versus 8% for random units", or Dice/IoU recovery for segmentation.
4. **Slice associations** (when metadata is given): which data groups the
   pattern is tied to.
5. **Offline HTML report** with the evidence. Vision tasks include paired
   dimmed-context and crop images.
6. **Optional plain-English narrative** from a local or cloud LLM. It describes
   evidence the math already produced and never chooses culprits. It is
   labeled as a hypothesis.

Illustrative output (numbers are made up):

```
Pattern A  (layer4, 6 channels)
  Failures fixed by ablation:  31/40 (78%)    random baseline: 8%
  Damage on clean samples:     1.2% accuracy drop
  Associated with:             supplier = "AcmeCo"  (34/40 failures)
```

## How it works

```
  [Trained model + failures + clean samples]
                    |
                    v
  +------------------------------------+
  | Adapter + HookManager              |  maps any model onto (Batch, Units,
  | (ANN, CNN, RNN, ViT, Mamba, ...)   |  Positions); hooks always cleaned up
  +------------------+-----------------+
                     v
  +------------------------------------+
  | 1. Triage (Grad x Act + activation |  one forward + backward pass narrows
  |    difference)                     |  hundreds of units to Top-K, K <= 12
  +------------------+-----------------+
                     v
  +------------------------------------+
  | 2. Causal proof (clean-mean        |  keeps only units that beat a random
  |    ablation + null control)        |  null; merges correlated units
  +------------------+-----------------+
                     v
  +------------------------------------+
  | 3. Report (terminal, HTML, JSON)   |  offline by default; optional LLM
  |    + optional LLM narrative        |  narrates the evidence
  +------------------------------------+
```

Details are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## What it is, and what it is not

BlameTrace **is** a failure-debugging tool. It finds and proves shortcut and
spurious-feature dependence behind a group of failures.

It **is not** a general explainability dashboard, a fairness auditor, or a
root-cause-in-the-data tool.

"Causal" here means the **interventional effect on model outputs**. A culprit
tells you where the model relies on a shortcut. A human still decides whether
the cause lives in the data, such as one supplier stamping a logo on photos.

Sometimes the honest result is "no proven culprit". If the failures are just
hard cases or noisy labels, no unit will pass the null-control test, and the
report says so.

## Planned model support

| Release | Adapter | Typical models | Planted-shortcut gate |
|---|---|---|---|
| v0.1 | `ANNAdapter` | MLPs, tabular nets | leaked column |
| v0.2 | `CNNAdapter` | ResNet, EfficientNet, ConvNeXt, VGG, MobileNet | corner watermark or glare |
| v0.3 | `RNNAdapter` | LSTM, GRU, BiLSTM, CRNN | boilerplate prefix |
| v0.4 | `TransformerAdapter` | BERT, GPT-2, LLaMA-style, ViT, Swin, CLIP towers | attention sink or corner patch |
| v0.5 (experimental) | `CNNMambaAdapter` | U-Mamba, VM-UNet | segmentation border artifact |
| any | `CustomAdapter` | any `nn.Module` with a layer list | n/a |
| any | `CompositeAdapter` | hybrids such as CNN + LSTM, CLIP | n/a |

Every adapter maps its model onto one tensor contract, `(Batch, Units,
Positions)`, so the core never needs to know the architecture.

Not supported: `torch.compile`, FSDP, DDP. Models run in eval mode on a single
device. Mixture-of-Experts and graph networks would go through `CustomAdapter`
with hints.

## Design guarantees

1. **One tensor contract.** Every adapter implements `to_canonical` and
   `from_canonical`, and non-ablated units round-trip bit-identical.
2. **No architecture branching in the core.** Model-family logic lives only in
   `adapters/`.
3. **Proof required.** No culprit is reported without a measured ablation
   score against a null.
4. **Offline first.** Everything works with `explainer=None` and zero network
   calls. Cloud LLM use is opt-in.
5. **Zero hook leaks.** Every hook runs inside `HookManager.session()` and is
   removed in a `finally:` block.

## Repository layout

```
src/blametrace/       package (docstring-only skeleton until Milestone 0)
tests/unit/           fast CPU tests
tests/benchmarks/     planted-shortcut release gates
docs/                 ARCHITECTURE, ROADMAP, adr/, rfcs/
.github/              CI, issue and pull request templates
FREEZE.md             scope contract: invariants, gates, change control
```

## Roadmap

Milestones are built strictly in order, and each ends with a planted-shortcut
benchmark that must pass before the next one starts.

| Milestone | Release | Focus |
|---|---|---|
| M0 | scaffold | hooks, contract, registry, CI |
| M1 | v0.1.0 | ANN / MLP |
| M1.x | v0.1.x | slice attribution, LLM explainer |
| M2 | v0.2.0 | CNN |
| M3 | v0.3.0 | RNN / LSTM / GRU |
| M4 | v0.4.0 | Transformers |
| v1.0 | v1.0.0 | API stabilization |
| M5 | v0.5.0 | hybrid CNN-Mamba, segmentation (experimental) |

Task checklists are in [`docs/ROADMAP.md`](docs/ROADMAP.md) and exact gate
conditions in [`FREEZE.md`](FREEZE.md).

## Development

```bash
git clone https://github.com/PradhyumnaAI/blametrace.git
cd blametrace
uv sync --all-extras --dev

uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest tests/unit
```

Requirements: Python 3.11 or newer, `torch>=2.3`, `pydantic>=2.7`. Optional
extras: `blametrace[llm]` for the explainer client, `blametrace[vision]` for
image evidence.

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`FREEZE.md`](FREEZE.md) first.
Ideas outside the frozen scope go through an RFC in
[`docs/rfcs/`](docs/rfcs/0000-template.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
