# JadeAgent Scale v2

## Purpose

This document defines a concrete scaling architecture for JadeAgent after the
current durable-runtime roadmap.

The goal is not "more agents" in the abstract. The goal is:

- support very large numbers of logical agents/tasks
- keep governance executable
- avoid global scans in the hot path
- make scheduling and recovery cheap enough to scale
- preserve tenant isolation, trust tiers, and memory boundaries
- remain simpler internally than an external interoperability protocol

This design assumes the current runtime baseline already exists:

- executable `NodeManifest`, `PolicyBundle`, `TaskPolicy`, `AccessGrant`
- durable `TaskStore`
- governed `MemoryRouter`
- routed mesh execution with tenant/trust/delegation filters
- audit events

Scale v2 is the next control-plane architecture on top of that base.

## Core Thesis

JadeAgent should not scale by turning every "agent" into a fully independent,
globally routable, constantly heartbeat-ing runtime entity.

That model becomes expensive too early.

Instead, JadeAgent Scale v2 should separate:

- logical agents: task identities, policies, memory scopes, sub-task trees
- physical workers: processes or nodes that actually execute model/tool work
- shard supervisors: control-plane owners for scheduling, leases, quotas, and
  local routing

This is the main scaling move.

The system should be optimized for:

- many logical agents/tasks
- fewer physical worker pools
- local scheduling inside shards
- minimal global coordination

## Non-Goals

Scale v2 does not try to solve:

- infinite horizontal scaling of LLM inference itself
- fully decentralized P2P operation without a coordination substrate
- global strong consistency for all memory and scheduling operations
- replacing the internal protocol with A2A

## Vocabulary

### Logical Agent

An execution identity with:

- task policy
- access grants
- memory scope
- tenant
- parent/child task lineage

It does not need to map 1:1 to a process.

### Physical Worker

A process or node that actually runs model/tool work.

### Shard

A partition of control-plane ownership. A shard owns:

- ready queues
- delayed/retry queues
- lease tracking
- local worker index
- local quota counters
- audit buffering

### Supervisor

The runtime loop responsible for one shard. It decides:

- task admission
- worker assignment
- retry/requeue
- local backpressure
- budget enforcement

## Topology

```text
                        +----------------------+
                        | Global Directory     |
                        | shard map            |
                        | tenant map           |
                        | budget summaries     |
                        +----------+-----------+
                                   |
                 +-----------------+-----------------+
                 |                                   |
      +----------v-----------+           +-----------v----------+
      | Shard Supervisor A   |           | Shard Supervisor B   |
      | tenant/capability    |           | tenant/capability    |
      | queues + leases      |           | queues + leases      |
      | local worker index   |           | local worker index   |
      +-----+-----------+----+           +-----+-----------+----+
            |           |                      |           |
   +--------v--+   +----v-------+     +-------v---+   +---v--------+
   | Worker A1 |   | Worker A2  |     | Worker B1 |   | Worker B2  |
   | model/tool|   | model/tool |     | model/tool|   | model/tool |
   +-----------+   +------------+     +-----------+   +------------+
```

## Why This Beats Global Mesh Scanning

The current router shape is still globally scan-oriented:

- `MeshRouter.route()` scans all nodes
- `DistributedMeshRouter._discover_nodes()` refreshes broad registry state
- `RedisTaskStore.requeue_expired()` scans tasks for expiry

That is acceptable for early distributed execution.
It is not the right hot path for very large fleets.

Scale v2 removes those global scans from the critical path by introducing
ownership and partitioning:

- task submission goes to exactly one shard owner
- lease expiry is tracked by that shard, not by broad scans
- worker selection happens inside the shard's local index
- audit aggregation is batched upward, not emitted as a global synchronous path

## Partitioning Strategy

### Shard Key

The primary shard key should be:

`tenant_id + capability`

Optional suffix:

`memory_scope` for very heavy workloads

This is the best first partition because:

- tenant isolation stays natural
- capability-local queues stay compact
- local scheduling can be capability-aware without a global search

### Shard Placement

Use rendezvous hashing or consistent hashing for shard owner selection.

The key point is this:

- task -> shard owner should be deterministic
- adding/removing supervisors should move only a fraction of keys

Do not use a global "find best worker in all nodes" path as the default.

## Control Plane

### Global Directory

The global directory should be small and slow-changing. It stores:

- shard ownership map
- tenant metadata
- policy version references
- global budget summaries
- supervisor liveness

It should not be asked to route every task deeply.

### Shard Supervisor

Each shard supervisor owns:

- `ready_queue`
- `retry_queue`
- `lease_wheel`
- `worker_heap`
- `quota_counters`
- `audit_buffer`
- `memory_scope_index`

This is where most scheduling work happens.

### Local Worker Index

Inside a shard, workers should be indexed by:

- capability
- trust tier
- available permits
- local queue depth
- health

This should be implemented as a local indexed structure, not a global scan.

Good candidates:

- heap per capability
- trust-bucketed heap
- token bucket / permit counters
- weighted fair queue for tenant budget enforcement

## Data Plane

### Worker Pools

Workers should be grouped by execution profile:

- LLM-heavy workers
- tool-heavy workers
- sandbox workers
- memory/index workers
- delegation/gateway workers

This is important because "one agent node type" becomes too vague at scale.

### Bounded Concurrency

Each physical worker should expose:

- `max_model_concurrency`
- `max_tool_concurrency`
- `max_memory_concurrency`
- `max_delegate_concurrency`

The scheduler should consume permits from the right bucket.

This is better than a single `max_inflight`.

## Async Runtime Model

Scale v2 should be async-native.

### Why

The current runtime is still mostly blocking:

- node step loop is synchronous
- delegation wait path polls
- Redis transport and task store are synchronous
- task execution occupies the node loop directly

This is fine for early correctness, but not for high concurrency.

### Required Runtime Shape

Introduce:

- `AsyncMeshTransport`
- `AsyncTaskStore`
- `AsyncMemoryStore`
- `AsyncMeshNode`
- `ShardSupervisor.run_forever()`

Each supervisor should use:

- `asyncio.TaskGroup`
- bounded `asyncio.Queue`
- semaphores for worker permits
- blocking-free transport polling
- deadline-aware timers instead of sweep loops

### Hot Loop

Each shard supervisor should have an event loop like:

1. accept new submissions
2. pull ready tasks
3. match tasks to local available workers
4. dispatch execution
5. renew or expire leases
6. flush audit/events in batch
7. update summaries

That loop should avoid sleeping blindly and should react to:

- queue wake-ups
- permit release
- deadline wheel ticks
- incoming result completions

## Queue Design

Each shard needs at least these queues:

- `ready_queue`
- `delayed_queue`
- `dead_letter_queue`
- `cancel_queue`

### Ready Queue

Partition by:

- priority
- tenant
- capability

Use weighted fair scheduling so one tenant cannot starve the shard.

### Delayed Queue

This holds:

- retries
- deferred tasks
- scheduled resumptions

Use a time-ordered structure:

- sorted set
- min-heap
- timing wheel

### Lease Tracking

Do not re-scan all running tasks.

Use a deadline-aware structure:

- hashed timing wheel
- min-heap by lease deadline
- bucketed expiry slots

This is one of the biggest scaling wins over the current broad sweep approach.

## Memory Architecture at Scale

Memory should remain split into three planes.

### 1. Private Buffer

Local to a worker or logical agent.

Use:

- in-process buffer
- optional local disk cache

No global coordination.

### 2. Task Scratchpad

Shared, temporary, lease-governed.

Use:

- append-only notes
- single-writer operational state
- shard-local ownership for critical state mutations

Operational state should be colocated with the shard that owns the task lease.

### 3. Semantic Shared Memory

Persistent and queryable.

Must be namespaced by:

- tenant
- memory_scope
- policy version if needed

Do not put operational task state here.

## Governance at Scale

Scale must not weaken governance.

Each task dispatch still carries:

- effective policy bundle
- effective access grants
- tenant id
- trust tier requirement
- memory scope

Supervisors should enforce:

- admission
- quota
- delegation allowlist
- trust-tier floor
- mount-level memory access class

Workers should enforce:

- tool/resource grants
- filesystem/network/shell/delegate restrictions
- memory mount rules

That means:

- shard supervisor = coarse control
- worker runtime = fine control

Both are required.

## Where Hillis-Steele Helps

Hillis-Steele is useful for parallel scan/reduction patterns.
It is not the scheduler.

### Good Uses

- aggregate queue depth summaries across worker groups
- compute cumulative tenant budget usage in parallel
- reduce audit counters across supervisors
- compute batch watermarks for checkpoint windows
- summarize health and pressure signals upward in logarithmic stages

### Bad Uses

- per-task worker routing
- lease ownership
- memory consistency
- task retry semantics
- delegation policy enforcement

So in Scale v2, Hillis-Steele belongs in the aggregation layer, not the task
placement core.

## Reduction Layer

Add a reduction plane above shard supervisors:

```text
workers -> shard supervisors -> regional reducers -> global directory
```

Reducers should batch and summarize:

- load
- token throughput
- queue pressure
- retry rates
- policy denials
- per-tenant budget consumption

This is where scan/reduction algorithms make sense.

## Complexity Targets

Scale v2 should aim for these hot-path properties:

- task submission: O(1) or O(log S) for shard lookup
- worker selection inside shard: O(log W_local)
- lease expiry: O(log R_local) or amortized O(1) with timing wheel
- retry insertion: O(log Q_local)
- global summaries: batched reduction, not O(N) global scans

Where:

- `S` = number of shards
- `W_local` = workers in a shard
- `R_local` = running tasks in a shard
- `Q_local` = queued tasks in a shard

This is the architectural difference between "distributed" and "scalable".

## Why This Could Outscale Classical Agent Frameworks

If implemented well, JadeAgent Scale v2 can be easier to scale than classical
agent frameworks because it is designed around:

- runtime ownership
- explicit scheduling
- durable leases
- executable governance
- task and memory separation
- control-plane reduction

Many classical agent frameworks are strongest at orchestration ergonomics.
Scale v2 is explicitly stronger when the problem becomes:

- many concurrent tasks
- tenant isolation
- risk-aware delegation
- runtime recovery
- observability and quotas

This is not automatically true today. It becomes true only if the architecture
below is actually built.

## Recommended Module Evolution

### Keep

- `jadeagent/governance.py`
- `jadeagent/mesh/protocol.py`
- `jadeagent/mesh/audit.py`
- `jadeagent/memory/router.py`

### Evolve

- `jadeagent/mesh/router.py`
  into shard-aware routing and local worker indexes
- `jadeagent/mesh/task_store.py`
  into async store + shard-owned lease tracking
- `jadeagent/mesh/node.py`
  into worker runtime + async supervisor runtime split
- `jadeagent/mesh/distributed_router.py`
  into shard directory and supervisor discovery

### Add

- `jadeagent/mesh/sharding.py`
- `jadeagent/mesh/supervisor.py`
- `jadeagent/mesh/async_task_store.py`
- `jadeagent/mesh/lease_wheel.py`
- `jadeagent/mesh/reducer.py`
- `jadeagent/mesh/worker_pool.py`

## Suggested Execution Model

### Ingress

1. Normalize task
2. Resolve effective policy
3. Compute shard key
4. Choose shard owner
5. Persist task into shard-owned queue

### Scheduling

1. Shard supervisor pops next eligible task
2. Budget and policy gate is checked
3. Local worker index selects a worker
4. Lease is created
5. Task is dispatched

### Completion

1. Worker returns result
2. Lease is closed
3. Scratchpad state is finalized
4. Audit batch is appended
5. Parent task or requester is notified

### Failure

1. Timeout/error/worker loss is detected
2. Lease expires through local deadline tracking
3. Retry policy is applied
4. Task returns to delayed or ready queue
5. Dead-letter path is used if attempts are exhausted

## Phased Build Plan

### Phase A - Async Foundation

- introduce `AsyncTaskStore`
- introduce `AsyncMeshTransport`
- add async worker runtime
- remove polling-based waiting in hot paths

### Phase B - Shard Supervisors

- add shard ownership map
- route by `tenant_id + capability`
- create local ready/retry queues per shard

### Phase C - Lease Wheel

- replace broad expiry sweep with deadline-aware lease tracking

### Phase D - Local Worker Index

- trust-tier buckets
- permit-aware local scheduling
- queue-pressure-aware worker scoring

### Phase E - Reduction Layer

- add shard summaries
- add reducer hierarchy
- add batch metrics and budget rollups
- use scan/reduction algorithms where beneficial

### Phase F - Gateway Layer

- A2A gateway
- external ingress adapters
- policy-safe public exposure

## Design Rule

The most important rule in Scale v2 is this:

Do not pay global coordination cost for work that can be decided inside a shard.

That single rule is what makes the architecture plausibly scalable.

## Bottom Line

JadeAgent should scale by becoming:

- async-native
- shard-owned
- supervisor-driven
- worker-pooled
- lease-indexed
- reduction-aware

Hillis-Steele helps only in the reduction-aware part.
It is useful, but it is not the architecture.
