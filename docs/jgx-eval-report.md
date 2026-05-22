# JadeAgent JGX Eval Report

- Suite: `core`
- Case runs: `6`
- Success rate: `1.0`
- Task completion rate: `1.0`
- Recovery success rate: `1.0`
- Duplicate tool executions: `0`
- Secret leak count: `0`
- Tokens prompt/completion/total: `392` / `12` / `404`
- Cost estimate USD: `0.0`
- Raw baseline p50 ms: raw `0.053` vs Jade+JGX `16.739`
- Runtime p50/p95 ms: `23.569` / `385.327`
- Restore p50/p95 ms: `0.981` / `0.981`

## Cases

| Case | Run id | Status | Runtime ms | Events | Snapshots | Key metric |
|---|---|---:|---:|---:|---:|---|
| `state_restore` | `eval_state_restore_20260515_161232_7a6f75ff_1` | `True` | 23.569 | 6 | 4 | restore_ms=0.981 |
| `tool_idempotency` | `eval_tool_idempotency_20260515_161232_7a6f75ff_1` | `True` | 70.662 | 20 | 14 | side_effect_calls=1; reused=1 |
| `crash_recovery` | `eval_crash_recovery_20260515_161232_7a6f75ff_1` | `True` | 35.458 | 8 | 6 | recovered=True; attempts=2 |
| `compatibility_guard` | `eval_compatibility_guard_20260515_161232_7a6f75ff_1` | `True` | 1.147 | 1 | 1 | blocked=1 |
| `raw_call_baseline` | `eval_raw_call_baseline_20260515_161232_7a6f75ff_1` | `True` | 19.515 | 6 | 4 | raw_ms=0.053; jade_ms=16.739; overhead_ms=16.686 |
| `mesh_project` | `eval_mesh_project_20260515_161232_7a6f75ff_1` | `True` | 385.327 | 1 | 1 | files=6; tests=True |

## Interpretation

These metrics are intentionally about reliability and governance: recovery,
idempotent side effects, restore compatibility, audit completeness, and
state overhead. They are designed for portfolio proof, not synthetic model
leaderboard scoring.
