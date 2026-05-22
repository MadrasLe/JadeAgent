"""
Mesh protocol primitives for distributed agent execution.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EnvelopeType(str, Enum):
    TASK = "task"
    RESULT = "result"
    HEARTBEAT = "heartbeat"
    CONTROL = "control"


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class MeshTask:
    capability: str
    prompt: str
    requester: str = "client"
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    priority: int = 0
    ttl: int = 8
    affinity: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    task_policy: dict[str, Any] = field(default_factory=dict)
    max_attempts: int = 3
    lease_seconds: float = 30.0
    tenant_id: str = ""
    memory_scope: str = ""
    parent_task_id: str | None = None
    min_trust_tier: str = "standard"


@dataclass
class TaskResult:
    task_id: str
    capability: str
    node_id: str
    state: TaskState = TaskState.COMPLETED
    output: str = ""
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.state == TaskState.COMPLETED

    def finalize(self, state: TaskState, output: str = "", error: str | None = None):
        self.state = state
        self.output = output
        self.error = error
        self.finished_at = time.time()


@dataclass
class MeshEnvelope:
    type: EnvelopeType
    source: str
    destination: str | None
    payload: dict[str, Any]
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str | None = None
    capability: str | None = None
    affinity: str | None = None
    priority: int = 0
    ttl: int = 8
    created_at: float = field(default_factory=time.time)
    trace: list[str] = field(default_factory=list)

    def hop(self, node_id: str) -> bool:
        self.trace.append(node_id)
        self.ttl -= 1
        return self.ttl >= 0

    @property
    def expired(self) -> bool:
        return self.ttl < 0


def envelope_to_dict(envelope: MeshEnvelope) -> dict[str, Any]:
    return {
        "type": envelope.type.value if isinstance(envelope.type, EnvelopeType) else str(envelope.type),
        "source": envelope.source,
        "destination": envelope.destination,
        "payload": dict(envelope.payload),
        "message_id": envelope.message_id,
        "task_id": envelope.task_id,
        "capability": envelope.capability,
        "affinity": envelope.affinity,
        "priority": envelope.priority,
        "ttl": envelope.ttl,
        "created_at": envelope.created_at,
        "trace": list(envelope.trace),
    }


def envelope_from_dict(data: dict[str, Any]) -> MeshEnvelope:
    env_type = data.get("type", EnvelopeType.CONTROL.value)
    try:
        env_type = EnvelopeType(env_type)
    except ValueError:
        env_type = EnvelopeType.CONTROL

    return MeshEnvelope(
        type=env_type,
        source=str(data.get("source", "")),
        destination=data.get("destination"),
        payload=dict(data.get("payload", {})),
        message_id=str(data.get("message_id", uuid.uuid4().hex)),
        task_id=data.get("task_id"),
        capability=data.get("capability"),
        affinity=data.get("affinity"),
        priority=int(data.get("priority", 0)),
        ttl=int(data.get("ttl", 8)),
        created_at=float(data.get("created_at", time.time())),
        trace=list(data.get("trace", [])),
    )


def make_task_envelope(task: MeshTask, source: str, destination: str | None = None) -> MeshEnvelope:
    return MeshEnvelope(
        type=EnvelopeType.TASK,
        source=source,
        destination=destination,
        task_id=task.task_id,
        capability=task.capability,
        affinity=task.affinity,
        priority=task.priority,
        ttl=task.ttl,
        payload={
            "prompt": task.prompt,
            "requester": task.requester,
            "metadata": dict(task.metadata),
            "task_policy": dict(task.task_policy),
            "max_attempts": task.max_attempts,
            "lease_seconds": task.lease_seconds,
            "tenant_id": task.tenant_id,
            "memory_scope": task.memory_scope,
            "parent_task_id": task.parent_task_id,
            "min_trust_tier": task.min_trust_tier,
        },
    )


def make_result_envelope(result: TaskResult, source: str, destination: str) -> MeshEnvelope:
    return MeshEnvelope(
        type=EnvelopeType.RESULT,
        source=source,
        destination=destination,
        task_id=result.task_id,
        capability=result.capability,
        priority=0,
        ttl=2,
        payload={
            "state": result.state.value,
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "node_id": result.node_id,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "metadata": dict(result.metadata),
        },
    )
