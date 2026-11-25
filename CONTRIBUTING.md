# Contributing to BlameTrace

Read [`FREEZE.md`](FREEZE.md) first. The scope, invariants, structure and
milestone order are frozen. Work outside them needs an RFC (see
[`docs/rfcs/`](docs/rfcs/0000-template.md)).

## 1. Local development setup

We use `uv` for deterministic environments and `ruff` for linting and formatting.

```bash
git clone https://github.com/PradhyumnaAI/blametrace.git
cd blametrace
uv sync --all-extras --dev
```

Run the full quality gate locally before opening a pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest tests/unit
uv run pytest tests/benchmarks -m benchmark
```

## 2. Branching and commit rules

- `main` is protected. Never push directly to `main`.
- Create short-lived branches named by milestone and scope:
  - `feat/m0-core-foundation`
  - `feat/m1-ann-adapter`
  - `fix/hook-tuple-cleanup`
  - `docs/adr-0004`
- Follow [Conventional Commits](https://www.conventionalcommits.org/):
  - `feat(cnn): add spatial cosine channel clustering`
  - `fix(hooks): guarantee handle removal when forward raises RuntimeError`
  - `test(ann): add planted proxy-column benchmark`

## 3. Rules for adding an architecture adapter

Every adapter in `src/blametrace/adapters/` must:

1. Subclass `BaseAdapter` and register through `@register_adapter("<name>")`.
2. Implement `discover_targets`, `to_canonical`, `from_canonical`, and
   `render_evidence`.
3. Never require changes to `src/blametrace/core/` for model-specific edge cases.
4. Ship with:
   - A round-trip test: `from_canonical(to_canonical(x), x)` equals `x` for
     every non-ablated unit.
   - A fast CPU planted-shortcut benchmark in `tests/benchmarks/` that runs in
     under 10 seconds and includes a negative control.

New adapters are proposed through the "New Architecture Adapter" issue template.

## 4. Pull request review bar

A pull request is ready to merge only when:

- All CI checks are green.
- It states which milestone and files it touches, and confirms the five
  invariants.
- Public functions have full type annotations and concise docstrings.
- Zero hook leaks are verified by tests.
- No placeholders or TODOs remain in merged code.
- Style: 88 columns, double quotes, sorted imports, `X | None` unions.

## 5. Working order

Milestones are built in order. Do not start Milestone N+1 while Milestone N's
planted-shortcut benchmark is failing.

## 6. Reporting a bug

Use the bug report template. Include a minimal model, the failure set (or a
synthetic stand-in), the BlameTrace version, the PyTorch version, and the
device.
