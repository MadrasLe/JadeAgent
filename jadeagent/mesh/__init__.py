"""Mesh orchestration primitives for distributed multi-agent execution."""

from .audit import AuditEvent, AuditSink
from .protocol import (
    EnvelopeType,
    MeshEnvelope,
    MeshTask,
    TaskResult,
    TaskState,
    envelope_to_dict,
    envelope_from_dict,
)
from .router import MeshRouter, NodeState
from .node import MeshNode, InMemoryMeshBus
from .transport import MeshTransport
from .async_transport import AsyncInMemoryMeshBus, AsyncMeshTransport, AsyncMeshTransportAdapter
from .async_task_store import AsyncInMemoryTaskStore, AsyncTaskStore, AsyncTaskStoreAdapter
from .async_node import AsyncMeshNode
from .lease_wheel import LeaseDeadlineIndex, LeaseRecord
from .reducer import (
    ReducerNode,
    ReductionSummary,
    ShardSummary,
    TenantBudgetSummary,
    hillis_steele_reduce,
    hillis_steele_scan,
)
from .sharding import ShardAssignment, ShardDirectory, SupervisorSpec
from .shard_runtime import ShardRuntime
from .supervisor import ShardSupervisor
from .worker_pool import LocalWorkerIndex, WorkerState
from .security import HMACSigner, ReplayConfig, ReplayProtector
from .distributed_router import DistributedMeshRouter
from .task_store import InMemoryTaskStore, RedisTaskStore, TaskRecord, TaskStore
from .agent_node import (
    MeshDelegationClient,
    extract_mesh_answer,
    make_agent_task_handler,
    make_mesh_delegate_tool,
)
from .async_agent_node import AsyncMeshDelegationClient, make_async_agent_task_handler

try:
    from .redis_transport import RedisMeshTransport
except Exception:  # pragma: no cover - optional dependency at runtime
    RedisMeshTransport = None

__all__ = [
    "AuditEvent",
    "AuditSink",
    "EnvelopeType",
    "MeshEnvelope",
    "MeshTask",
    "TaskResult",
    "TaskState",
    "envelope_to_dict",
    "envelope_from_dict",
    "MeshRouter",
    "DistributedMeshRouter",
    "NodeState",
    "MeshNode",
    "InMemoryMeshBus",
    "AsyncMeshNode",
    "AsyncMeshTransport",
    "AsyncMeshTransportAdapter",
    "AsyncInMemoryMeshBus",
    "LeaseRecord",
    "LeaseDeadlineIndex",
    "LocalWorkerIndex",
    "ShardSummary",
    "TenantBudgetSummary",
    "ReductionSummary",
    "ReducerNode",
    "hillis_steele_scan",
    "hillis_steele_reduce",
    "ShardDirectory",
    "SupervisorSpec",
    "ShardAssignment",
    "ShardRuntime",
    "ShardSupervisor",
    "WorkerState",
    "TaskStore",
    "TaskRecord",
    "InMemoryTaskStore",
    "RedisTaskStore",
    "AsyncTaskStore",
    "AsyncTaskStoreAdapter",
    "AsyncInMemoryTaskStore",
    "MeshTransport",
    "HMACSigner",
    "ReplayConfig",
    "ReplayProtector",
    "RedisMeshTransport",
    "MeshDelegationClient",
    "AsyncMeshDelegationClient",
    "extract_mesh_answer",
    "make_agent_task_handler",
    "make_async_agent_task_handler",
    "make_mesh_delegate_tool",
]
