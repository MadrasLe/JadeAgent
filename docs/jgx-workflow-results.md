# JGX Workflow Results

Date: 2026-05-07

This note records a full local workflow run using the OpenRouter-compatible
backend, the requested Nemotron model, tool calls, graph orchestration, mesh
task execution, and JGX restore.

## Run

- Workflow id: `20260507_183609`
- Model: `nvidia/nemotron-3-super-120b-a12b:free`
- Runtime output root:
  `C:\Users\gabri\AppData\Local\Temp\jadeagent_jgx_workflow_20260507_183609`
- Result: `ok`

## Workflow Shape

1. `jgx_workflow_analyzer` used the real OpenRouter model to analyze why JGX
   state capsules are useful for agent workflows.
2. `jgx_workflow_scorer` used the `compute_jgx_score` tool three times to score
   candidate JGX use cases.
3. A `StateGraph` collected, ranked, and annotated the scoring results while
   writing graph checkpoints.
4. A `MeshNode` reviewed the graph result and wrote mesh task lifecycle
   checkpoints.
5. A fresh agent restored the scorer run from its JGX capsule and recovered a
   `COMPLETED` snapshot with 12 messages.
6. `jgx_workflow_summarizer` used the real model to summarize the workflow
   result.

## JGX Capsules

| Capsule | Run id | Phase | Snapshots | Events |
|---|---:|---:|---:|---:|
| Analyzer agent | `3e5dc87e21f24776929471810461cb9a` | `COMPLETED` | 4 | 6 |
| Scorer agent | `4d2f039a925b47db9b88243bf4110074` | `COMPLETED` | 15 | 17 |
| Summary agent | `49c72e1b9a594bc287986f700aac2d5c` | `COMPLETED` | 4 | 6 |
| Graph workflow | `graph_20260507_183609` | `COMPLETED` | 4 | 5 |
| Mesh task | `mesh_20260507_183609` | `COMPLETED` | 2 | 3 |

## Scoring Result

The model successfully used tool calling. `compute_jgx_score` was called three
times.

| Candidate | Impact | Evidence | Effort | Risk | Score |
|---|---:|---:|---:|---:|---:|
| `restore-after-failure` | 9 | 9 | 4 | 3 | 8.48 |
| `governed-audit-trail` | 8 | 9 | 3 | 2 | 8.23 |
| `mesh-worker-migration` | 10 | 8 | 7 | 5 | 8.06 |

Top candidate:

```text
restore-after-failure
```

Graph annotation:

```text
Top candidate restore-after-failure scored 8.48; JGX value is strongest when it prevents lost work and proves execution history.
```

Mesh review:

```json
{"mesh_review":"ok","top_candidate":"restore-after-failure","top_score":8.48,"prompt_seen":"Validate ranked JGX workflow result"}
```

## What This Proved

- JGX works with a real OpenRouter LLM backend, not only fake test backends.
- Agent runs persist `manifest.json`, `events.jsonl`, and checkpoint snapshots.
- Tool calls are captured inside the scorer run. The scorer capsule had 15
  snapshots and 17 events.
- Graph execution writes state-machine checkpoints independently of agent
  checkpoints.
- Mesh task lifecycle state is captured as its own JGX capsule.
- Restore is practical: a fresh agent restored the scorer run at `COMPLETED`
  with 12 messages.

## Caveat

The mesh leg used the in-memory mesh bus without a durable `TaskStore`, so the
mesh emitted the existing legacy durability warning. This does not invalidate
the JGX checkpoint proof, but it means the next production-grade demo should run
mesh with a durable task store as well as a state store.
