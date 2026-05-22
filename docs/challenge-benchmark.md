# Adversarial Challenge Benchmark

The earlier benchmarks answer useful but easier questions:

- runtime overhead for a deterministic graph;
- objective output quality for small scripted answers;
- JGX eval metrics for recovery and idempotency.

The adversarial challenge benchmark asks a harder portfolio question:

> What happens when the task attacks common agent-framework weak spots?

Run it with:

```bash
python benchmarks/challenge_compare.py --out-dir benchmarks/out --json
```

Install LangGraph first to enable the optional comparison target:

```bash
pip install -q langgraph
```

## What It Tests

Targets:

- `raw_plain`: direct Python functions with no durable runtime layer.
- `langgraph_plain`: LangGraph `StateGraph` without a custom checkpointer,
  idempotency ledger, or audit capsule.
- `jade_agent_jgx`: JadeAgent plus SQLite-backed JGX state.

Cases:

- `schema_contract`: strict JSON-compatible contract and dependency order.
- `conflict_resolution`: detect incompatible requirements instead of pretending
  they are satisfiable.
- `replay_side_effect`: replay the same request without duplicating an external
  side effect.
- `crash_recovery`: resume from a checkpoint after a simulated crash.
- `audit_evidence`: prove the run with events, snapshots, integrity hash, and
  secret-leak checks.

## How To Read It

`schema_contract` and `conflict_resolution` are output-quality checks. A good
workflow runner with the same deterministic backend should pass them.

The stronger signal is in:

- `replay_side_effect`;
- `crash_recovery`;
- `audit_evidence`.

Those cases test runtime guarantees, not prose quality. This is where JGX is
supposed to differentiate: durable checkpoints, idempotency records, replay
evidence, event history, snapshot history, and capsule verification.

## Claim Boundary

This benchmark is intentionally named `langgraph_plain`, not just `langgraph`.
It does not claim LangGraph cannot implement these guarantees. A production
LangGraph app can add checkpointing, persistence, custom idempotency keys, and
audit logging.

The useful claim is narrower and stronger:

> JadeAgent JGX exposes these concerns as first-class runtime behavior, while a
> plain graph orchestration baseline needs additional engineering to match the
> same reliability and audit surface.

## Colab

After mounting Drive:

```python
%cd /content/drive/MyDrive/JadeAgent
!pip install -q -e .
!pip install -q langgraph
!python benchmarks/challenge_compare.py --out-dir benchmarks/out --json
```

Show the newest report:

```python
from pathlib import Path
reports = sorted(Path("benchmarks/out").glob("challenge_compare_*.md"))
print(reports[-1])
print(reports[-1].read_text())
```
