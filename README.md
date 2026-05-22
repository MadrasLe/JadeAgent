# JadeAgent

JadeAgent is an agent runtime for governed execution: ReAct agents, tools,
sessions, graph workflows, mesh tasks, and durable JGX state capsules.

The project started around MegaGemm/local inference, but the current core focus
is broader: reliable agent execution with checkpoints, recovery, idempotent tool
replay, state inspection, and audit evidence.

## Highlights

- OpenAI-compatible backend support for OpenAI, Groq, OpenRouter, Ollama,
  Together, and similar APIs.
- Optional MegaGemm local backend for local GPU inference experiments.
- `@tool` decorator with JSON-schema generation from type hints.
- ReAct-style agent loop with tool calling and session snapshots.
- Graph orchestration with optional JGX checkpointing.
- Mesh task primitives for local and distributed worker experiments.
- JGX governed execution state: manifests, events, snapshots, compatibility
  fingerprints, integrity verification, and timeline export.
- CLI tools for state inspection, history, export, verification, timelines,
  crash-recovery demos, and deterministic eval reports.
- Deterministic benchmark suite comparing raw Python, JadeGraph, JadeGraph+JGX,
  LangGraph plain, and LangGraph durable SQLite.

## Installation

```bash
cd JadeAgent
pip install -e .
```

Optional extras:

```bash
# Local MegaGemm backend
pip install -e ".[local]"

# Memory dependencies
pip install -e ".[memory]"

# Redis mesh transport
pip install -e ".[network]"

# Sandbox provider integration
pip install -e ".[sandbox]"

# Everything defined by the project
pip install -e ".[all]"
```

Optional benchmark dependencies for LangGraph comparisons:

```bash
pip install langgraph
pip install "langgraph-checkpoint-sqlite==2.0.11"
```

## Quick Start

### Simple Agent

```python
from jadeagent import Agent
from jadeagent.backends import OpenAICompatBackend

backend = OpenAICompatBackend(
    model="gpt-4o-mini",
    api_key="sk-...",
)

agent = Agent(backend=backend, name="jade")
print(agent.chat("Explain durable agent state in one paragraph."))
```

### Agent With A Tool

```python
from jadeagent import Agent, tool
from jadeagent.backends import OpenAICompatBackend


@tool(description="Calculate a math expression")
def calculator(expression: str) -> str:
    return str(eval(expression))


agent = Agent(
    backend=OpenAICompatBackend("gpt-4o-mini", api_key="sk-..."),
    tools=[calculator],
    verbose=False,
)

result = agent.run("What is 15% of 340?")
print(result.answer)
print(result.tool_calls_made)
```

### Agent With JGX State

```python
from jadeagent import Agent, SqliteStateStore
from jadeagent.backends import OpenAICompatBackend

store = SqliteStateStore("examples/generated/eval/state.sqlite3")
agent = Agent(
    backend=OpenAICompatBackend("gpt-4o-mini", api_key="sk-..."),
    state_store=store,
    run_id="demo_run",
    verbose=False,
)

result = agent.run("Create a short implementation plan.")
print(result.answer)
print(store.inspect("demo_run"))
store.close()
```

## JGX State, Recovery, And Audit

JGX means Jade Governed eXecution. It treats an agent, graph, or mesh task run as
an inspectable state machine:

- manifest: identity and restore-compatibility metadata;
- events: append-only record of runtime activity;
- snapshots: restorable execution checkpoints;
- integrity: event-chain hashes, snapshot hashes, and secret-leak checks;
- tooling: CLI inspect/history/latest/export/timeline/verify commands.

Useful commands:

```bash
jade demo crash-recovery
jade demo mesh-code-project

jade state inspect <run_id> --store examples/generated/eval/state.sqlite3
jade state history <run_id> --store examples/generated/eval/state.sqlite3
jade state latest <run_id> --store examples/generated/eval/state.sqlite3
jade state timeline <run_id> --store examples/generated/eval/state.sqlite3 --html timeline.html
jade state verify <run_id> --store examples/generated/eval/state.sqlite3

jade eval run --suite core --runs 1
jade eval report --suite core
```

Core eval coverage:

- state restore;
- tool idempotency;
- crash recovery;
- restore compatibility guard;
- raw-call baseline overhead;
- token usage estimates;
- task completion status;
- mesh-generated Python project with unit tests.

## Benchmarks

The benchmarks are deterministic and local by default. They are meant to prove
runtime behavior, not live model intelligence.

### Portfolio Overview

Run the consolidated report:

```bash
python benchmarks/portfolio_overview.py --out-dir benchmarks/out --runtime-runs 25 --json
```

It combines:

- runtime comparison;
- controlled quality comparison;
- adversarial challenge comparison;
- production-style LangGraph durable comparison;
- JGX reliability eval.

### Runtime: Raw Python vs JadeGraph vs LangGraph

```bash
python benchmarks/langgraph_compare.py --runs 25 --out-dir benchmarks/out --json
```

This measures framework overhead on the same deterministic five-node workflow.
JadeGraph without state is intentionally lightweight; JadeGraph+JGX adds
SQLite-backed governed state and therefore adds latency.

### Controlled Quality

```bash
python benchmarks/quality_compare.py --out-dir benchmarks/out --json
```

This holds the backend output constant and scores objective rubric quality.
When the scripted backend is controlled, JadeAgent and LangGraph should tie on
answer quality. Framework orchestration does not make the model smarter by
itself.

### Adversarial Challenge

```bash
python benchmarks/challenge_compare.py --out-dir benchmarks/out --json
```

This compares raw Python, `langgraph_plain`, and `jade_agent_jgx` on strict
contracts, conflicting requirements, side-effect replay, crash recovery, and
audit evidence. It shows the gap between plain graph orchestration and a runtime
with first-class durable state.

### LangGraph Durable SQLite

```bash
python benchmarks/durable_compare.py --out-dir benchmarks/out --json
```

This is the fairer LangGraph comparison. The LangGraph target uses SQLite
checkpointing, `thread_id`, `durability="sync"`, `@task`, and recovery with
`invoke(None, config)`.

Current honest reading:

> Properly configured LangGraph durable SQLite matches JGX on recovery and
> idempotent side-effect replay. JGX differentiates by adding a portable governed
> execution capsule with event-level audit, snapshots, integrity verification,
> and event-chain hashes.

## Architecture Map

```text
jadeagent/
  backends/    LLM provider backends
  core/        Agent, session, tools, streaming, result types
  graph/       StateGraph workflow runtime
  mesh/        Mesh task routing, stores, workers, async runtime
  state/       JGX manifests, events, snapshots, stores, integrity
  council/     Debate, MoA, Tree-of-Thought style experiments
  memory/      Buffer/router/memory primitives
  sandbox/     Local/E2B sandbox providers
  skills/      Skill library and generation helpers
```

## Documentation

- [Architecture](docs/architecture.md)
- [JGX State Capsules](docs/jgx-state-capsules.md)
- [JGX Eval Metrics](docs/jgx-eval-metrics.md)
- [Benchmark Overview](docs/benchmark-overview.md)
- [LangGraph Colab Benchmark](docs/langgraph-colab-benchmark.md)
- [LangGraph Durable Benchmark](docs/langgraph-durable-benchmark.md)
- [Challenge Benchmark](docs/challenge-benchmark.md)
- [Quality Benchmark](docs/quality-benchmark.md)
- [Mesh Runtime](docs/mesh-runtime.md)
- [Governance](docs/governance.md)
- [Scale Blueprint](docs/jadeagent-scale-v2.md)

## Claim Boundary

The benchmark suite supports a strong but specific claim:

> JadeAgent/JGX is a governed agent runtime that matches controlled graph
> baselines on deterministic output quality and provides first-class durable
> execution evidence: idempotent replay, crash recovery, state inspection,
> integrity verification, and event-level audit.

It does not prove that JadeAgent is universally better than LangGraph or that it
improves model intelligence. For live model quality claims, add real model runs,
hidden tasks, and human or LLM judging.

## License

MIT
