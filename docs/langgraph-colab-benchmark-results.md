# LangGraph Colab Benchmark Results

Date: 2026-05-19

Environment: Google Colab, JadeAgent installed from Google Drive with
`pip install -e .`, LangGraph installed with `pip install langgraph`.

Command:

```bash
python benchmarks/langgraph_compare.py --runs 25 --out-dir benchmarks/out --json
```

## Summary

| Target | Runs | Completion | p50 ms | p95 ms | Avg ms | Notes |
|---|---:|---:|---:|---:|---:|---|
| `raw_python` | 25 | 1.0 | 0.012589 | 0.037536 | 0.015487 | lower-bound Python baseline |
| `jade_graph` | 25 | 1.0 | 0.019482 | 0.050221 | 0.112991 | overhead vs raw p50: 0.006893 ms |
| `jade_graph_jgx` | 25 | 1.0 | 20.95733 | 26.370523 | 21.973276 | events=175, snapshots=150, verify=True |
| `langgraph` | 25 | 1.0 | 3.257988 | 13.132446 | 6.052755 | overhead vs raw p50: 3.245399 ms |

## Interpretation

This benchmark measures deterministic workflow runtime overhead, not model
quality or reasoning ability.

The local JadeGraph runtime was close to raw Python on this tiny five-node
workflow. LangGraph had higher runtime overhead, which is expected for a mature
external framework with broader abstractions. JadeGraph+JGX was much slower
because it writes SQLite-backed governed state: manifests, events, snapshots,
and integrity-verifiable execution history.

The portfolio claim supported by this result is:

> JadeGraph without durable state is lightweight for simple deterministic
> workflows. JGX adds measurable persistence overhead, but buys governed
> execution state, event history, snapshots, and integrity verification.

## Boundary

This result should not be presented as a general claim that JadeAgent is
universally faster than LangGraph. It is a microbenchmark for one deterministic
workflow. Quality, recovery behavior, LLM output quality, tool use, and
production reliability need separate tests.
