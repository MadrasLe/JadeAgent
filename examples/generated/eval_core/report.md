# JadeAgent JGX Eval Report

- Suite: `core`
- Case runs: `5`
- Success rate: `1.0`
- Recovery success rate: `0.2`
- Duplicate tool executions: `0`
- Secret leak count: `0`
- Runtime p50/p95 ms: `51.43` / `475.534`
- Restore p50/p95 ms: `0.918` / `0.918`

## Cases

| Case | Run id | Status | Runtime ms | Events | Snapshots | Key metric |
|---|---|---:|---:|---:|---:|---|
| `state_restore` | `eval_state_restore_20260514_232838_39d0441a_1` | `True` | 23.374 | 6 | 4 | restore_ms=0.918 |
| `tool_idempotency` | `eval_tool_idempotency_20260514_232838_39d0441a_1` | `True` | 76.455 | 20 | 14 | side_effect_calls=1; reused=1 |
| `crash_recovery` | `eval_crash_recovery_20260514_232838_39d0441a_1` | `True` | 51.43 | 8 | 6 | recovered=True; attempts=2 |
| `compatibility_guard` | `eval_compatibility_guard_20260514_232838_39d0441a_1` | `True` | 4.571 | 1 | 1 | blocked=1 |
| `mesh_project` | `eval_mesh_project_20260514_232838_39d0441a_1` | `True` | 475.534 | 1 | 1 | files=6; tests=True |

## Interpretation

These metrics are intentionally about reliability and governance: recovery,
idempotent side effects, restore compatibility, audit completeness, and
state overhead. They are designed for portfolio proof, not synthetic model
leaderboard scoring.
