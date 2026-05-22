# JadeAgent JGX Eval Report

- Suite: `core`
- Case runs: `5`
- Success rate: `1.0`
- Recovery success rate: `1.0`
- Duplicate tool executions: `0`
- Secret leak count: `0`
- Runtime p50/p95 ms: `14.493` / `58.589`
- Restore p50/p95 ms: `0.918` / `0.918`

## Cases

| Case | Run id | Status | Runtime ms | Events | Snapshots | Key metric |
|---|---|---:|---:|---:|---:|---|
| `compatibility_guard` | `eval_compatibility_guard_20260514_232838_39d0441a_1` | `True` | 1.122 | 1 | 1 | blocked=1 |
| `crash_recovery` | `eval_crash_recovery_20260514_232838_39d0441a_1` | `True` | 25.624 | 8 | 6 | recovered=True; attempts=2 |
| `mesh_project` | `eval_mesh_project_20260514_232838_39d0441a_1` | `True` | 1.414 | 1 | 1 | files=6; tests=True |
| `state_restore` | `eval_state_restore_20260514_232838_39d0441a_1` | `True` | 14.493 | 6 | 4 | restore_ms=0.918 |
| `tool_idempotency` | `eval_tool_idempotency_20260514_232838_39d0441a_1` | `True` | 58.589 | 20 | 14 | side_effect_calls=1; reused=1 |

## Interpretation

These metrics are intentionally about reliability and governance: recovery,
idempotent side effects, restore compatibility, audit completeness, and
state overhead. They are designed for portfolio proof, not synthetic model
leaderboard scoring.
