# JadeAgent Compared With Other Agent Frameworks

This comparison is intentionally architectural rather than marketing-oriented.

## Positioning

JadeAgent is not trying to be only a nicer wrapper for model calls. Its stronger
position is:

> a governed agent runtime with mesh execution, durable JGX state, checkpointing,
> CLI inspection, and crash recovery.

## Comparison

| Framework | Strong At | JadeAgent Differentiator |
|---|---|---|
| LangGraph | graph workflows, persistence, controllable agent flows | JGX state spans agent, graph, mesh, governance, and CLI inspection |
| CrewAI | ergonomic teams/roles/tasks | JadeAgent focuses on runtime durability and policy enforcement |
| AutoGen / Microsoft Agent Framework | multi-agent conversations and orchestration | JadeAgent emphasizes state capsules and governed recovery |
| LlamaIndex | data/RAG-centric agents | JadeAgent is data-source agnostic and runtime-centric |
| OpenAI Agents SDK | provider integration, sessions, tracing | JadeAgent is provider-independent and exposes local state stores |
| Temporal | durable workflow execution | JadeAgent borrows durable-execution ideas but targets LLM agents/tools |

## Where JadeAgent Is Strong Now

- executable governance;
- tool policy enforcement;
- mesh task execution;
- JGX checkpoints;
- SQLite state store;
- state CLI;
- crash recovery demo;
- medium mesh code project demo;
- idempotent tool result reuse.

## Where JadeAgent Still Needs Work

- stronger replay semantics for every runtime phase;
- production distributed state/task stores;
- signed capsules and integrity checks;
- timeline visualization;
- public packaging polish;
- broader concurrency tests.

## Best Portfolio Framing

JadeAgent should be presented as an agent infrastructure project:

```text
Built a Python agent runtime with executable governance, mesh-based execution,
durable JGX state capsules, SQLite checkpointing, CLI inspection, idempotent
tool replay, and crash recovery demos.
```

