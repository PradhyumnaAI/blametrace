# ADR-0003: The LLM narrates evidence and never selects culprits

## Status
Accepted

## Context
Stage 3 can send evidence to a local or cloud LLM or VLM to produce
plain-English diagnoses. Language models can state plausible causes that the
evidence does not support, and cloud calls can leak sensitive data.

## Decision
- Culprits, patterns and recovery numbers come only from Stages 1 and 2.
- The explainer receives compact evidence and returns JSON validated against a
  strict Pydantic schema.
- Anything the explainer cites must exist in the evidence; unsupported claims
  are rejected.
- Output is HTML-escaped and labeled as a hypothesis.
- Cloud calls are opt-in. With `explainer=None` the library makes no network
  calls.
- Requests are limited to 3-5 concurrent calls and cached by evidence hash in a
  local SQLite file.
- `httpx` is installed only with the `blametrace[llm]` extra.

## Consequences
- Reports are complete and trustworthy offline.
- The narrative layer can be swapped or removed without changing results.
- The offline invariant is testable by blocking sockets in the test suite.
