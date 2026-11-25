# ADR-0002: Causal proof requires a null control

## Status
Accepted

## Context
Failure samples sit near decision boundaries, so ablating almost any unit flips
some of them. A raw flip rate therefore proves little. Single-unit ablation can
also miss redundancy, where two units compensate for each other.

## Decision
Every reported culprit carries:

1. `flip_rate` and `confidence_drop` from clean-mean ablation on failures.
2. The same measures for an equal number of random units from the same layer,
   reported as a z-score or percentile against that null.
3. Collateral damage: the effect of ablating the culprit on clean samples.
4. For each `FailurePattern`, a joint-ablation score as well as per-unit scores.

A unit that does not beat the null, or that damages clean samples as much as
failures, is not reported as a culprit.

## Consequences
- Culprit lists are shorter and more defensible.
- "No proven culprit" is a valid, reported outcome.
- Every benchmark includes a negative-control model trained without the
  shortcut.
- "Causal" in this project means interventional effect on model outputs, not
  root cause in the data.
