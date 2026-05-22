# LangGraph Durable Benchmark

This benchmark compares JadeAgent JGX against a production-style LangGraph
configuration, not the plain LangGraph baseline.

Run:

```bash
python benchmarks/durable_compare.py --out-dir benchmarks/out --json
```

The LangGraph target uses:

- `langgraph-checkpoint-sqlite`;
- `SqliteSaver`;
- a stable `thread_id`;
- `durability="sync"`;
- `@task` for side effects;
- `invoke(None, config)` for recovery after failure.

Those details matter. LangGraph durable recovery is not the same as simply
calling the graph again with the original input. For failure recovery, the
documented pattern is to resume with the same `thread_id` and `None` input.

Official docs:

- <https://docs.langchain.com/oss/python/langgraph/durable-execution>
- <https://docs.langchain.com/oss/python/langgraph/persistence>
- <https://reference.langchain.com/python/langgraph/graph/state/StateGraph/compile>

## What It Tests

- `side_effect_resume`: crash after a side-effect task result and resume without
  re-executing the side effect.
- `artifact_crash_recovery`: persist phase 1, crash, resume, and complete phase
  2 exactly once.
- `state_history`: store enough durable history for later inspection.
- `audit_depth`: compare JGX-style event-level audit evidence: events,
  snapshots, integrity verification, and event-chain hash.

## How To Read It

This benchmark is intentionally fairer to LangGraph than `challenge_compare`.
When configured correctly, LangGraph should perform well on recovery and
idempotent task replay.

The expected JGX differentiation is narrower:

> LangGraph durable provides checkpoint-based recovery. JGX provides recovery
> plus a portable governed execution capsule with event-level audit and
> integrity verification.

So the most honest reading is not "LangGraph cannot do durability." It can.
The question is whether you want checkpoint recovery alone, or checkpoint
recovery plus first-class governed execution evidence.

## Colab

```python
%cd /content/drive/MyDrive/JadeAgent
!pip install -q -e .
!pip install -q "langgraph-checkpoint-sqlite==2.0.11"
!python benchmarks/durable_compare.py --out-dir benchmarks/out --json
```
