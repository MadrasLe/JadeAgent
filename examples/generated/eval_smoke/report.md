# JadeAgent JGX Eval Report

- Suite: `fast`
- Case runs: `2`
- Success rate: `1.0`
- Recovery success rate: `0.0`
- Duplicate tool executions: `0`
- Secret leak count: `0`
- Runtime p50/p95 ms: `30.157` / `75.153`
- Restore p50/p95 ms: `1.375` / `1.375`

## Cases

| Case | Run id | Status | Runtime ms | Events | Snapshots | Key metric |
|---|---|---:|---:|---:|---:|---|
| `state_restore` | `eval_state_restore_20260514_232753_413f6861_1` | `True` | 30.157 | 6 | 4 | restore_ms=1.375 |
| `tool_idempotency` | `eval_tool_idempotency_20260514_232753_413f6861_1` | `True` | 75.153 | 20 | 14 | side_effect_calls=1; reused=1 |

## Interpretation

These metrics are intentionally about reliability and governance: recovery,
idempotent side effects, restore compatibility, audit completeness, and
state overhead. They are designed for portfolio proof, not synthetic model
leaderboard scoring.
