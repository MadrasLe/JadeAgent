# Mesh Runtime

The JadeAgent mesh layer lets agents and handlers execute work as routed tasks.
It supports local in-memory demos today and has durable task store primitives for
production-oriented paths.

## Core Types

- `MeshTask`: capability, prompt, requester, metadata, policy, attempts, lease.
- `TaskResult`: task output, failure state, metadata.
- `MeshNode`: synchronous worker/coordinator.
- `AsyncMeshNode`: async worker/coordinator.
- `MeshRouter`: capability routing and liveness.
- `TaskStore`: durable task lifecycle abstraction.

## Lifecycle

```text
submit -> pending -> claimed/running -> completed
                              |
                              -> failed -> retry pending
                              |
                              -> failed final
```

Workers claim tasks by capability. A claim grants a lease. If a worker does not
finish, the task store can requeue the task until the retry budget is exhausted.

## JGX Integration

When a mesh node receives a `state_store`, each mesh task can write a JGX run.
The current checkpoints are:

- `RUNNING`
- `COMPLETED`
- `FAILED`

Task metadata may define the run id:

```python
MeshTask(
    capability="recover_document",
    prompt="...",
    metadata={"jgx_run_id": "my_run"},
)
```

Without `jgx_run_id`, the default is `mesh_<task_id>`.

## Crash Recovery Demo

`examples/jgx_crash_recovery.py` proves the recovery path:

1. worker A writes a checkpoint;
2. worker A crashes;
3. task is retried;
4. worker B reads the SQLite JGX state;
5. worker B completes from the checkpoint.

CLI inspection then shows the timeline:

```bash
jade state history <run_id> --store state.sqlite3
```

## Production Gap

For production, mesh needs tighter coupling between `TaskStore` and
`StateStore`:

- task lease recovery should locate the latest compatible JGX snapshot;
- task retries should pass the intended JGX run id;
- idempotency records should cover external tools and activity handlers;
- distributed stores should be tested under concurrent workers.

