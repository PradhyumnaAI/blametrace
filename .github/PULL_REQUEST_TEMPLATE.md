## Summary
<!-- What does this PR change, and which milestone/issue does it close? -->
Closes #

Milestone: M_
Files touched:

## Freeze check
- [ ] Change is inside the scope of `FREEZE.md`, or links an accepted RFC

## Invariants check
- [ ] No architecture-specific `if/else` added inside `src/blametrace/core/`
- [ ] Canonical `(B, U, P)` round-trip shape parity tested, non-ablated units bit-identical
- [ ] Hooks only inside `HookManager.session()`, zero leaks verified on exception
- [ ] Every reported culprit carries a measured ablation score against a null
- [ ] Works fully offline (`explainer=None`), no network calls on the default path

## Verification
- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass
- [ ] `uv run mypy` passes
- [ ] `uv run pytest tests/unit` passes
- [ ] `uv run pytest tests/benchmarks -m benchmark` passes (paste metrics below if applicable)
