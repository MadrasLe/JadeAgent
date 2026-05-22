"""Restorable runtime snapshots for Jade governed execution capsules."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.types import Message, Role, ToolCall


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    return repr(value)


def tool_call_to_dict(tool_call: ToolCall | None) -> dict[str, Any] | None:
    if tool_call is None:
        return None
    return {
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": _json_safe(tool_call.arguments),
    }


def tool_call_from_dict(data: dict[str, Any] | None) -> ToolCall | None:
    if not data:
        return None
    return ToolCall(
        id=str(data.get("id", "")),
        name=str(data.get("name", "")),
        arguments=dict(data.get("arguments", {})),
    )


def message_to_snapshot(message: Message) -> dict[str, Any]:
    role = message.role.value if isinstance(message.role, Role) else str(message.role)
    return {
        "role": role,
        "content": message.content,
        "tool_calls": [
            tool_call_to_dict(tool_call)
            for tool_call in (message.tool_calls or [])
            if tool_call is not None
        ],
        "tool_call_id": message.tool_call_id,
        "name": message.name,
    }


def message_from_snapshot(data: dict[str, Any]) -> Message:
    tool_calls: list[ToolCall] = []
    for raw_call in data.get("tool_calls") or []:
        if not raw_call:
            continue
        if "function" in raw_call:
            function = raw_call.get("function") or {}
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}
            raw_call = {
                "id": raw_call.get("id", ""),
                "name": function.get("name", ""),
                "arguments": arguments,
            }
        tool_call = tool_call_from_dict(raw_call)
        if tool_call is not None:
            tool_calls.append(tool_call)

    return Message(
        role=str(data.get("role", Role.USER.value)),
        content=data.get("content"),
        tool_calls=tool_calls or None,
        tool_call_id=data.get("tool_call_id"),
        name=data.get("name"),
    )


@dataclass
class SessionSnapshot:
    """Conversation state that can be restored into a Session."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    backend: str = ""
    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "backend": self.backend,
            "messages": _json_safe(self.messages),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionSnapshot":
        return cls(
            snapshot_id=str(data.get("snapshot_id") or uuid.uuid4().hex),
            created_at=float(data.get("created_at", time.time())),
            backend=str(data.get("backend", "")),
            messages=list(data.get("messages", [])),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_messages(
        cls,
        messages: list[Message],
        *,
        backend: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "SessionSnapshot":
        return cls(
            backend=backend,
            messages=[message_to_snapshot(message) for message in messages],
            metadata=dict(metadata or {}),
        )

    def restore_messages(self) -> list[Message]:
        return [message_from_snapshot(message) for message in self.messages]


@dataclass
class GraphRuntimeSnapshot:
    current_node: str = ""
    next_nodes: list[str] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_node": self.current_node,
            "next_nodes": list(self.next_nodes),
            "variables": _json_safe(self.variables),
            "outputs": _json_safe(self.outputs),
            "iteration": self.iteration,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GraphRuntimeSnapshot | None":
        if not data:
            return None
        return cls(
            current_node=str(data.get("current_node", "")),
            next_nodes=[str(node) for node in data.get("next_nodes", [])],
            variables=dict(data.get("variables", {})),
            outputs=dict(data.get("outputs", {})),
            iteration=int(data.get("iteration", 0)),
        )


@dataclass
class MeshRuntimeSnapshot:
    shard_key: str = ""
    lease_owner: str = ""
    lease_deadline: float = 0.0
    attempt: int = 0
    task_state: str = ""
    task_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_key": self.shard_key,
            "lease_owner": self.lease_owner,
            "lease_deadline": self.lease_deadline,
            "attempt": self.attempt,
            "task_state": self.task_state,
            "task_metadata": _json_safe(self.task_metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MeshRuntimeSnapshot | None":
        if not data:
            return None
        return cls(
            shard_key=str(data.get("shard_key", "")),
            lease_owner=str(data.get("lease_owner", "")),
            lease_deadline=float(data.get("lease_deadline", 0.0)),
            attempt=int(data.get("attempt", 0)),
            task_state=str(data.get("task_state", "")),
            task_metadata=dict(data.get("task_metadata", {})),
        )


@dataclass
class AgentRuntimeSnapshot:
    """Restorable agent state at a safe execution boundary."""

    phase: str = "NEW"
    step: int = 0
    session: SessionSnapshot = field(default_factory=SessionSnapshot)
    plan: list[str] = field(default_factory=list)
    pending_tool_call: dict[str, Any] | None = None
    last_observation: dict[str, Any] = field(default_factory=dict)
    graph: GraphRuntimeSnapshot | None = None
    mesh: MeshRuntimeSnapshot | None = None
    memory_refs: list[str] = field(default_factory=list)
    model_state_ref: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.session.messages

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "phase": self.phase,
            "step": self.step,
            "session": self.session.to_dict(),
            "plan": list(self.plan),
            "pending_tool_call": _json_safe(self.pending_tool_call),
            "last_observation": _json_safe(self.last_observation),
            "graph": self.graph.to_dict() if self.graph is not None else None,
            "mesh": self.mesh.to_dict() if self.mesh is not None else None,
            "memory_refs": list(self.memory_refs),
            "model_state_ref": _json_safe(self.model_state_ref),
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentRuntimeSnapshot":
        session_data = data.get("session") or {}
        if "messages" in data and "messages" not in session_data:
            session_data = {"messages": data.get("messages", [])}
        return cls(
            snapshot_id=str(data.get("snapshot_id") or uuid.uuid4().hex),
            created_at=float(data.get("created_at", time.time())),
            phase=str(data.get("phase", "NEW")),
            step=int(data.get("step", 0)),
            session=SessionSnapshot.from_dict(session_data),
            plan=[str(item) for item in data.get("plan", [])],
            pending_tool_call=data.get("pending_tool_call"),
            last_observation=dict(data.get("last_observation", {})),
            graph=GraphRuntimeSnapshot.from_dict(data.get("graph")),
            mesh=MeshRuntimeSnapshot.from_dict(data.get("mesh")),
            memory_refs=[str(ref) for ref in data.get("memory_refs", [])],
            model_state_ref=data.get("model_state_ref"),
            metadata=dict(data.get("metadata", {})),
        )
