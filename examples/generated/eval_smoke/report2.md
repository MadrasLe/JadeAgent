# JadeAgent JGX Eval Report

- Suite: `fast`
- Case runs: `2`
- Success rate: `1.0`
- Recovery success rate: `0.0`
- Duplicate tool executions: `0`
- Secret leak count: `0`
- Runtime p50/p95 ms: `18.001` / `59.001`
- Restore p50/p95 ms: `0.0` / `0.0`

## Cases

| Case | Run id | Status | Runtime ms | Events | Snapshots | Key metric |
|---|---|---:|---:|---:|---:|---|
| `state_restore` | `eval_state_restore_20260514_232753_413f6861_1` | `True` | 18.001 | 6 | 4 | restore_ms= |
| `tool_idempotency` | `eval_tool_idempotency_20260514_232753_413f6861_1` | `True` | 59.001 | 20 | 14 | side_effect_calls=; reused=1 |

## Interpretation

These metrics are intentionally about reliability and governance: recovery,
idempotent side effects, restore compatibility, audit completeness, and
state overhead. They are designed for portfolio proof, not synthetic model
leaderboard scoring.
