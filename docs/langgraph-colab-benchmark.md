# LangGraph Colab Benchmark

This benchmark compares the same deterministic five-node workflow across:

- `raw_python`: plain Python function calls.
- `jade_graph`: JadeAgent graph execution without durable state.
- `jade_graph_jgx`: JadeAgent graph execution with SQLite-backed JGX state.
- `langgraph`: LangGraph `StateGraph`, when `langgraph` is installed.

The benchmark is intentionally deterministic. It measures runtime/framework
overhead and state/audit capability, not LLM intelligence.

## Colab Cells

Run these cells in Google Colab after putting JadeAgent in Drive.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Adjust the path if your Drive folder has a different name:

```python
%cd /content/drive/MyDrive/JadeAgent
```

Install runtime dependencies:

```python
!pip install -q -e .
!pip install -q langgraph
```

Run the benchmark:

```python
!python benchmarks/langgraph_compare.py --runs 25 --out-dir benchmarks/out --json
```

Show the newest markdown report:

```python
from pathlib import Path
reports = sorted(Path("benchmarks/out").glob("langgraph_compare_*.md"))
print(reports[-1])
print(reports[-1].read_text())
```

Download the JSON and Markdown artifacts if desired:

```python
from google.colab import files
for path in sorted(Path("benchmarks/out").glob("langgraph_compare_*"))[-2:]:
    files.download(str(path))
```

## What To Look For

- `completion_rate`: all targets should complete the same deterministic task.
- `duration_ms_p50` and `duration_ms_p95`: runtime overhead per target.
- `overhead_vs_raw_p50`: extra p50 latency compared to plain Python.
- `jgx_event_count` and `jgx_snapshot_count`: audit/state volume from JGX.
- `jgx_verify_ok`: whether exported JGX state passes integrity checks.

## Interpretation

LangGraph is a mature graph runtime. JadeAgent should not claim to beat it on
raw graph scheduling alone. The JadeAgent argument is different:

> The extra JGX overhead buys durable governed execution state, event hashes,
> snapshots, restore inspection, and auditability.

If `jade_graph` is close to `langgraph`, the graph engine is competitive for
simple local workflows. If `jade_graph_jgx` is slower, that is expected; the
question is whether the overhead is acceptable for the audit/recovery benefits.
