# JGX Eval Metrics

JadeAgent now has a deterministic eval layer for portfolio-grade proof. The
goal is not to benchmark model intelligence. The goal is to prove that agent
execution is governed, durable, inspectable, and recoverable.

## Commands

```bash
jade eval run --suite core --runs 1
jade eval report --suite core
```

Useful state inspection commands:

```bash
jade state timeline <run_id> --store examples/generated/eval/state.sqlite3 --html timeline.html
jade state verify <run_id> --store examples/generated/eval/state.sqlite3
```

## Suites

- `fast`: state restore and tool idempotency.
- `reliability`: fast suite plus crash recovery and compatibility guard.
- `core`: reliability suite plus a raw-call baseline and a mesh-generated
  Python project that runs unit tests.

## Metrics

- `success_rate`: completed cases divided by total cases.
- `task_completion_rate`: cases whose primary task reached a completed state.
- `recovery_success_rate`: crash-recovery cases that completed after a simulated
  failure.
- `duplicate_tool_executions`: repeated side-effectful tool executions after a
  replay attempt.
- `prompt_tokens`, `completion_tokens`, and `total_tokens`: backend usage when
  the backend reports it, or deterministic local estimates in scripted evals.
- `cost_estimate_usd`: cost estimate when a price table is available. Scripted
  local evals use `0.0` because no provider is billed.
- `raw_runtime_ms`, `jade_runtime_ms`, and `state_overhead_ms`: direct backend
  call compared with Agent+JGX for the `raw_call_baseline` case.
- `restore_latency_ms`: time to load the latest snapshot and restore a session.
- `event_count` and `snapshot_count`: JGX audit density per run.
- `secret_leak_count`: likely API keys or tokens found in manifest, events, or
  snapshots.
- `event_chain_hash`: deterministic hash over the ordered event log.
- `unit_tests_passed`: whether the generated mesh project passed its own tests.

## Portfolio Narrative

The strongest claim is:

> JadeAgent does not only call LLMs. It models agent work as governed execution
> state, records checkpoints, avoids duplicate tool side effects, survives
> crashes, blocks incompatible restore, and emits metrics that can be audited.

That is the distinction from a plain agent wrapper.
