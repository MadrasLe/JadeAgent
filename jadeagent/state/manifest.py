"""Manifest primitives for Jade governed execution (.jgx) capsules."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


JGX_FORMAT = "jade-governed-execution"
JGX_MAGIC = "JGX1"
JGX_SCHEMA_VERSION = "1.0"


def _json_default(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, (set, tuple)):
        return list(value)
    return repr(value)


def canonical_json_hash(value: Any) -> str:
    """Return a stable sha256 hash for JSON-compatible runtime metadata."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fingerprint_mapping(value: dict[str, Any] | None) -> str:
    """Hash a mapping, returning an empty string for empty metadata."""

    if not value:
        return ""
    return canonical_json_hash(value)


@dataclass
class JadeStateManifest:
    """Compatibility and identity metadata for a .jgx execution capsule."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str = ""
    agent_id: str = ""
    tenant_id: str = ""
    capability: str = ""
    parent_run_id: str = ""
    state_kind: str = "agent"
    backend: str = ""
    model_fingerprint: str = ""
    policy_hash: str = ""
    tool_registry_hash: str = ""
    memory_scope_hash: str = ""
    payload_hash: str = ""
    latest_snapshot_id: str = ""
    format: str = JGX_FORMAT
    magic: str = JGX_MAGIC
    schema_version: str = JGX_SCHEMA_VERSION
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "magic": self.magic,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "capability": self.capability,
            "parent_run_id": self.parent_run_id,
            "state_kind": self.state_kind,
            "backend": self.backend,
            "model_fingerprint": self.model_fingerprint,
            "policy_hash": self.policy_hash,
            "tool_registry_hash": self.tool_registry_hash,
            "memory_scope_hash": self.memory_scope_hash,
            "payload_hash": self.payload_hash,
            "latest_snapshot_id": self.latest_snapshot_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JadeStateManifest":
        return cls(
            format=str(data.get("format", JGX_FORMAT)),
            magic=str(data.get("magic", JGX_MAGIC)),
            schema_version=str(data.get("schema_version", JGX_SCHEMA_VERSION)),
            run_id=str(data.get("run_id") or uuid.uuid4().hex),
            task_id=str(data.get("task_id", "")),
            agent_id=str(data.get("agent_id", "")),
            tenant_id=str(data.get("tenant_id", "")),
            capability=str(data.get("capability", "")),
            parent_run_id=str(data.get("parent_run_id", "")),
            state_kind=str(data.get("state_kind", "agent")),
            backend=str(data.get("backend", "")),
            model_fingerprint=str(data.get("model_fingerprint", "")),
            policy_hash=str(data.get("policy_hash", "")),
            tool_registry_hash=str(data.get("tool_registry_hash", "")),
            memory_scope_hash=str(data.get("memory_scope_hash", "")),
            payload_hash=str(data.get("payload_hash", "")),
            latest_snapshot_id=str(data.get("latest_snapshot_id", "")),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            metadata=dict(data.get("metadata", {})),
        )

    @property
    def capsule_hash(self) -> str:
        """Hash the manifest fields that define restore compatibility."""

        return canonical_json_hash({
            "format": self.format,
            "schema_version": self.schema_version,
            "tenant_id": self.tenant_id,
            "capability": self.capability,
            "state_kind": self.state_kind,
            "backend": self.backend,
            "model_fingerprint": self.model_fingerprint,
            "policy_hash": self.policy_hash,
            "tool_registry_hash": self.tool_registry_hash,
            "memory_scope_hash": self.memory_scope_hash,
        })
