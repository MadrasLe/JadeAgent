# JGX Crash Recovery Results

Date: 2026-05-07

This demo simulates a worker crash after a checkpoint and resumes the
same mesh task from a SQLite-backed JGX state store.

## Run

- Workflow id: `20260507_223229`
- Run id: `crash_recovery_20260507_223229`
- SQLite state: `C:\Users\gabri\JadeAgent\examples\generated\jgx_crash_recovery_20260507_223229\state.sqlite3`
- Output artifact: `C:\Users\gabri\JadeAgent\examples\generated\jgx_crash_recovery_20260507_223229\recovered_artifact.txt`
- Final task state: `completed`
- Result state: `completed`

## Attempts

- `worker_before_crash` -> `crashed_after_checkpoint`
- `worker_after_recovery` -> `resumed_and_completed`

## JGX Inspect

- Latest phase: `COMPLETED`
- Snapshots: `6`
- Events: `9`

## Event Types

```text
demo_started
mesh_task_started
checkpoint
simulated_crash
checkpoint
mesh_task_started
checkpoint
recovered_and_completed
checkpoint
```

## Artifact

```text
phase 1: draft written before crash
phase 2: recovered worker completed artifact
```
