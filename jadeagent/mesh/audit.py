"""
Audit primitives for distributed mesh execution.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class AuditEvent:
    event_type: str
    task_id: str = ""
    node_id: str = ""
    tenant_id: str = ""
    parent_task_id: str = ""
    tool_name: str = ""
    resource: str = ""
    action: str = ""
    scope: str = ""
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "task_id": self.task_id,
            "node_id": self.node_id,
            "tenant_id": self.tenant_id,
            "parent_task_id": self.parent_task_id,
            "tool_name": self.tool_name,
            "resource": self.resource,
            "action": self.action,
            "scope": self.scope,
            "message": self.message,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEvent:
        metadata = data.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {"raw": metadata}
        if not isinstance(metadata, dict):
            metadata = {}
        return cls(
            event_type=str(data.get("event_type", "")),
            task_id=str(data.get("task_id", "")),
            node_id=str(data.get("node_id", "")),
            tenant_id=str(data.get("tenant_id", "")),
            parent_task_id=str(data.get("parent_task_id", "")),
            tool_name=str(data.get("tool_name", "")),
            resource=str(data.get("resource", "")),
            action=str(data.get("action", "")),
            scope=str(data.get("scope", "")),
            message=str(data.get("message", "")),
            metadata=metadata,
            created_at=float(data.get("created_at", time.time())),
        )


class AuditSink(Protocol):
    def record_event(self, event: AuditEvent | dict[str, Any]):
        ...

    def list_events(self, task_id: str | None = None, limit: int = 100) -> list[AuditEvent]:
        ...


def coerce_audit_event(event: AuditEvent | dict[str, Any]) -> AuditEvent:
    if isinstance(event, AuditEvent):
        return event
    return AuditEvent.from_dict(dict(event))
