# JadeAgent Architecture

JadeAgent is a Python runtime for agent execution. Its core idea is that agents
should not be treated as temporary Python loops only. They should be executable,
governed, inspectable state machines.

## Layers

```text
Agent API
  Session, tools, ReAct loop, streaming

Governance
  NodeManifest, PolicyBundle, TaskPolicy, resource checks, audit hooks

State
  JGX manifests, events, snapshots, FileStateStore, SqliteStateStore

Graph
  StateGraph, node transitions, optional checkpointing

Mesh
  MeshNode, AsyncMeshNode, routing, task stores, leases, worker pools

Backends
  OpenAI-compatible providers, MegaGemm integration hooks
```

## Execution Model

An `Agent.run()` call follows a bounded loop:

1. checkpoint `PLANNING`;
2. call the model;
3. checkpoint `OBSERVING`;
4. checkpoint `READY_TOOL` before each tool;
5. execute or replay the tool result;
6. checkpoint observation;
7. finish with `COMPLETED` or `FAILED`.

When a `StateStore` is attached, the run becomes a JGX state machine with:

- a `JadeStateManifest`;
- append-only `JadeStateEvent` records;
- restorable `AgentRuntimeSnapshot` objects.

## JGX State

JGX means **Jade Governed eXecution**. It captures execution state, not semantic
memory. A JGX run records enough information to inspect, audit, restore, export,
or recover a task.

Supported stores:

- `FileStateStore`: transparent `.jgx` directory capsules.
- `SqliteStateStore`: durable single-file state store for local production and
  demos.
- `InMemoryStateStore`: tests and embedded usage.

## Tool Idempotency

Tool calls can produce irreversible side effects. JadeAgent now records tool
results as JGX events keyed by:

```text
run_id + step + tool_call_id + tool_name + arguments
```

If the same tool call reappears in the same run, JadeAgent reuses the recorded
result instead of executing the tool again. This is the foundation for safe
replay.

## CLI

The `jade` CLI exposes state inspection:

```bash
jade state list --store .jade_state.sqlite3
jade state inspect <run_id> --store .jade_state.sqlite3
jade state history <run_id> --store .jade_state.sqlite3
jade state latest <run_id> --store .jade_state.sqlite3
jade state export <run_id> --store .jade_state.sqlite3 --out run.jgx
jade state timeline <run_id> --store .jade_state.sqlite3 --html timeline.html
jade state verify <run_id> --store .jade_state.sqlite3
```

It also exposes demos:

```bash
jade demo crash-recovery
jade demo mesh-code-project
```

And deterministic eval suites:

```bash
jade eval run --suite core --runs 1
jade eval report --suite core
```

The eval layer is intentionally local-first. It measures recovery, restore,
tool idempotency, compatibility blocking, generated project tests, audit
completeness, event/snapshot volume, secret hygiene, and runtime overhead.

## Current Boundaries

JadeAgent currently supports checkpointing and basic replay safety. The next
hard boundary is full continuation semantics for every phase:

- resume before model call;
- resume after model call but before tool;
- resume after tool result;
- resume inside graph node;
- resume mesh tasks after lease recovery.
