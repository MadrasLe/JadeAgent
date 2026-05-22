# JGX State Capsules

JGX means **Jade Governed eXecution**. A `.jgx` capsule is the agent-side
equivalent of MGX state, but for governed execution instead of inference cache.

The design goal is simple:

> JadeAgent turns agents into portable, governed state machines.

## What JGX Stores

A `.jgx` run is a transparent directory capsule:

```text
<run_id>.jgx/
  manifest.json
  events.jsonl
  snapshots/
    <snapshot_id>.json
  payloads/
```

The first implementation is local-first and JSON-first on purpose. It gives the
runtime a stable contract before adding Redis, binary packing, signatures, or
remote artifact stores.

## Concepts

State is not memory. State says where execution is and what can resume. Memory
is what the agent knows or can retrieve. Artifacts are large outputs or files.
Audit explains why actions were allowed or denied.

JGX keeps those surfaces separate:

- `manifest.json`: identity, schema, tenant, capability, policy hash, tool
  registry hash, memory scope hash, backend fingerprint.
- `events.jsonl`: append-only state machine transitions and runtime records.
- `snapshots/*.json`: restorable checkpoints captured at safe boundaries.
- `payloads/`: optional content-addressed blobs for large artifacts.

## Agent Checkpoints

`Agent` accepts a `state_store` and writes checkpoints during `run()`:

- `PLANNING`
- `AWAITING_MODEL`
- `OBSERVING`
- `READY_TOOL`
- `COMPLETED`
- `FAILED`

Example:

```python
from jadeagent import Agent, FileStateStore

store = FileStateStore(".jade_state")
agent = Agent(backend=backend, tools=[...], state_store=store)

result = agent.run("Do the task")
print(agent.run_id)
print(store.inspect(agent.run_id))
```

For local durable state in a single file, use SQLite:

```python
from jadeagent import Agent, SqliteStateStore

store = SqliteStateStore(".jade_state.sqlite3")
agent = Agent(backend=backend, state_store=store)
```

Restore the latest session snapshot:

```python
agent.restore_state(run_id)
```

Restore validates compatibility gates such as tenant, backend, and tool
registry when the manifest contains those fingerprints.

## Session Snapshots

`Session` now supports:

```python
snapshot = session.snapshot()
session.restore_snapshot(snapshot)
restored = Session.restore(backend, snapshot)
```

The snapshot preserves messages, tool calls, and tool results. It does not own
semantic memory or external artifacts.

## Graph Checkpoints

`CompiledGraph.run()` remains backward compatible:

```python
result = graph.compile().run({"value": 1})
```

To make a graph execution durable:

```python
result = graph.compile().run(
    {"value": 1},
    state_store=store,
    run_id="graph_run",
)
```

The graph writes snapshots with the current node, next node, merged variables,
and iteration number.

## Mesh Task Checkpoints

`MeshNode` and `AsyncMeshNode` accept an optional `state_store`. When present,
task lifecycle state is captured as `.jgx` too:

```python
worker = MeshNode(
    node_id="worker",
    capabilities={"summarize"},
    router=router,
    bus=bus,
    task_handler=handler,
    state_store=store,
)
```

Mesh phases currently include:

- `RUNNING`
- `COMPLETED`
- `FAILED`

The mesh snapshot stores task id, capability, tenant, memory scope, lease owner,
lease deadline, attempt number, output, and error. A task can choose its capsule
id with `task.metadata["jgx_run_id"]`; otherwise the store uses
`mesh_<task_id>`.

## CLI

The `jade` CLI can inspect both directory-backed and SQLite-backed state stores.

```bash
jade state list --store .jade_state
jade state inspect <run_id> --store .jade_state
jade state history <run_id> --store .jade_state
jade state latest <run_id> --store .jade_state
jade state export <run_id> --store .jade_state --out exported.jgx
jade state timeline <run_id> --store .jade_state --html timeline.html
jade state verify <run_id> --store .jade_state
jade state list --store .jade_state.sqlite3
jade state inspect <run_id> --store .jade_state.sqlite3
jade state history <run_id> --store .jade_state.sqlite3 --limit 50 --json
jade state latest <run_id> --store .jade_state.sqlite3 --json
jade state export <run_id> --store .jade_state.sqlite3 --out exported.jgx
jade state timeline <run_id> --store .jade_state.sqlite3 --json
jade state verify <run_id> --store .jade_state.sqlite3 --json
```

When `--store-type` is omitted, paths ending in `.db`, `.sqlite`, or
`.sqlite3` use `SqliteStateStore`; other paths use `FileStateStore`.

Runnable demos are exposed through CLI as well:

```bash
jade demo crash-recovery
jade demo mesh-code-project
```

## Integrity And Timeline

`jade state verify` computes a deterministic chained event hash, snapshot hash,
schema checks, latest-snapshot checks, and a secret hygiene scan. It reports the
paths where likely API keys or tokens appear without printing the secret value.

`jade state timeline` merges events and snapshots into one chronological view.
With `--html`, it writes a small standalone timeline report that is useful for
debugging, demos, and portfolio screenshots.

## Eval Suites

JadeAgent includes deterministic local eval suites for proving the JGX runtime:

```bash
jade eval run --suite core --runs 1
jade eval report --suite core
```

The `core` suite checks state restore, tool idempotency, crash recovery,
restore compatibility guards, a raw backend-call baseline, and a mesh-generated
Python project with unit tests. The report focuses on reliability metrics:
success rate, task completion rate, token usage, raw-vs-JGX overhead, recovery
rate, duplicate tool executions, event/snapshot counts, secret leak count,
runtime, and restore latency.

## Tool Idempotency

Tool results are recorded as JGX events with an idempotency key derived from:

```text
run_id + step + tool_call_id + tool_name + arguments
```

If a later replay of the same run produces the same key, JadeAgent emits a
`tool_result_reused` event and returns the previously recorded result instead
of executing the tool again. This prevents duplicate side effects for replayed
tool calls.

## Restore Rules

A runtime should only restore a JGX capsule when the current execution context
is compatible:

- tenant scope matches;
- policy hash is accepted;
- tool registry is compatible;
- memory scope is compatible;
- backend and model fingerprints match when model state is present;
- schema version is supported.

This is the main distinction from plain conversation history. JGX is not just a
chat transcript. It is governed execution state.

## MGX Relationship

MGX remains the right layer for model/inference machine state such as KV cache,
tokenizer hash, model hash, and compiled tensor payloads.

JGX may reference MGX when a backend supports model-state persistence, but JGX
does not depend on it:

```text
.jgx = governed agent execution state
.mgx = model/inference machine state
```

No Prophet layer is included in JGX core. Reuse and speculative inference belong
to model-serving paths, not the agent state-machine contract.
