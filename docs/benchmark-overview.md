# Benchmark Overview

JadeAgent now has a consolidated benchmark runner:

```bash
python benchmarks/portfolio_overview.py --out-dir benchmarks/out --json
```

It creates one portfolio-oriented report that links the detailed artifacts from:

- runtime comparison: raw Python, JadeGraph, JadeGraph+JGX, and LangGraph when
  installed;
- controlled quality comparison: deterministic objective rubric;
- adversarial challenge comparison: replay, crash recovery, and audit evidence;
- production-style LangGraph durable comparison: SQLite checkpointer,
  `thread_id`, `durability="sync"`, `@task`, and resume with `None`;
- JGX reliability eval: restore, idempotency, crash recovery, compatibility,
  raw-call overhead, and mesh project generation.

## Why This Exists

Single benchmarks are easy to overread. The overview keeps the claims separated:

- runtime answers "how much overhead does this add?";
- controlled quality answers "does the framework preserve a good structured
  output when the backend is held constant?";
- adversarial capability answers "does the runtime survive replay, crash, and
  audit requirements?";
- durable LangGraph comparison answers "what changes when LangGraph is
  configured properly for persistence and side-effect replay?";
- eval reliability answers "does JGX behave consistently across its own core
  reliability cases?".

## Recommended Portfolio Claim

The strongest honest claim is:

> JadeAgent/JGX matches controlled output-quality baselines when the backend is
> held constant, while adding first-class runtime evidence for idempotent tool
> replay, crash recovery, state inspection, and integrity verification.

The more precise LangGraph comparison is:

> Properly configured LangGraph durable SQLite matches JGX on recovery and
> side-effect replay; JGX differentiates on portable governed execution
> evidence: event-level audit, capsule verification, and chain hashes.

Avoid claiming that deterministic local benchmarks prove model intelligence.
For that, add live model runs, hidden tasks, and a human or LLM judge.

## Colab

```python
%cd /content/drive/MyDrive/JadeAgent
!pip install -q -e .
!pip install -q langgraph
!python benchmarks/portfolio_overview.py --out-dir benchmarks/out --runtime-runs 25 --json
```

Show the newest report:

```python
from pathlib import Path
reports = sorted(Path("benchmarks/out").glob("portfolio_overview_*/portfolio_overview_*.md"))
print(reports[-1])
print(reports[-1].read_text())
```
