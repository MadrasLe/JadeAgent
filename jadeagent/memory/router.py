"""
Shared memory routing and storage primitives for mesh nodes.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from .buffer import BufferMemory
from ..governance import MemoryMount, NodeManifest, ResourceRequirement, TaskPolicy, check_access, memory_mount_allowed


class MemoryStore(ABC):
    @abstractmethod
    def append_note(self, task_id: str, mount_name: str, note: str, node_id: str, metadata: dict | None = None):
        ...

    @abstractmethod
    def list_notes(self, task_id: str, mount_name: str, limit: int = 100) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def write_state(self, task_id: str, mount_name: str, state: dict[str, Any], node_id: str):
        ...

    @abstractmethod
    def read_state(self, task_id: str, mount_name: str) -> dict[str, Any]:
        ...


class InMemorySharedMemoryStore(MemoryStore):
    def __init__(self):
        self._notes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self._state: dict[tuple[str, str], dict[str, Any]] = {}

    def append_note(self, task_id: str, mount_name: str, note: str, node_id: str, metadata: dict | None = None):
        self._notes[(task_id, mount_name)].append({
            "node_id": node_id,
            "note": note,
            "metadata": dict(metadata or {}),
        })

    def list_notes(self, task_id: str, mount_name: str, limit: int = 100) -> list[dict[str, Any]]:
        return list(self._notes[(task_id, mount_name)][-limit:])

    def write_state(self, task_id: str, mount_name: str, state: dict[str, Any], node_id: str):
        self._state[(task_id, mount_name)] = {
            "node_id": node_id,
            "state": dict(state),
        }

    def read_state(self, task_id: str, mount_name: str) -> dict[str, Any]:
        data = self._state.get((task_id, mount_name), {})
        if not data:
            return {}
        return dict(data.get("state", {}))


class RedisMemoryStore(MemoryStore):
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "jade:memory",
        tls: bool = False,
        tls_ca_certs: str | None = None,
        tls_certfile: str | None = None,
        tls_keyfile: str | None = None,
        tls_cert_reqs: str | None = "required",
        redis_kwargs: dict[str, Any] | None = None,
    ):
        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "RedisMemoryStore requires redis package. Install with: pip install redis"
            ) from exc

        kwargs = dict(redis_kwargs or {})
        if tls:
            kwargs.setdefault("ssl", True)
            if tls_ca_certs:
                kwargs["ssl_ca_certs"] = tls_ca_certs
            if tls_certfile:
                kwargs["ssl_certfile"] = tls_certfile
            if tls_keyfile:
                kwargs["ssl_keyfile"] = tls_keyfile
            if tls_cert_reqs:
                kwargs["ssl_cert_reqs"] = tls_cert_reqs

        self._client = redis.Redis.from_url(redis_url, decode_responses=True, **kwargs)
        self._client.ping()
        self.key_prefix = key_prefix.rstrip(":")

    def _notes_key(self, task_id: str, mount_name: str) -> str:
        return f"{self.key_prefix}:notes:{task_id}:{mount_name}"

    def _state_key(self, task_id: str, mount_name: str) -> str:
        return f"{self.key_prefix}:state:{task_id}:{mount_name}"

    def append_note(self, task_id: str, mount_name: str, note: str, node_id: str, metadata: dict | None = None):
        self._client.rpush(
            self._notes_key(task_id, mount_name),
            json.dumps({"node_id": node_id, "note": note, "metadata": dict(metadata or {})}, separators=(",", ":")),
        )

    def list_notes(self, task_id: str, mount_name: str, limit: int = 100) -> list[dict[str, Any]]:
        values = self._client.lrange(self._notes_key(task_id, mount_name), max(0, -limit), -1)
        return [json.loads(value) for value in values]

    def write_state(self, task_id: str, mount_name: str, state: dict[str, Any], node_id: str):
        self._client.hset(self._state_key(task_id, mount_name), mapping={
            "node_id": node_id,
            "state": json.dumps(dict(state), separators=(",", ":")),
        })

    def read_state(self, task_id: str, mount_name: str) -> dict[str, Any]:
        raw = self._client.hget(self._state_key(task_id, mount_name), "state")
        if not raw:
            return {}
        return json.loads(raw)


class MemoryRouter:
    def __init__(
        self,
        shared_store: MemoryStore | None = None,
        task_store=None,
        semantic_factory=None,
        audit_sink: Any = None,
    ):
        self.shared_store = shared_store or InMemorySharedMemoryStore()
        self.task_store = task_store
        self.semantic_factory = semantic_factory or self._default_semantic_factory
        self.audit_sink = audit_sink
        self._private_buffers: dict[tuple[str, str], BufferMemory] = {}
        self._semantic_memories: dict[tuple[str, str], Any] = {}

    def _default_semantic_factory(self, tenant_id: str, memory_scope: str):
        from .shorestone import ShoreStoneMemory

        key = f"{tenant_id or 'default'}_{memory_scope or 'default'}".replace("/", "_")
        return ShoreStoneMemory(collection=f"jade_{key}")

    def _emit(self, event_type: str, **payload):
        if self.audit_sink is None:
            return
        record_fn = getattr(self.audit_sink, "record_event", None)
        if callable(record_fn):
            record_fn({
                "event_type": event_type,
                **payload,
            })

    def _mount_for(self, node_manifest: NodeManifest, mount_name: str) -> MemoryMount:
        for mount in node_manifest.memory_mounts:
            if mount.name == mount_name:
                return mount
        raise KeyError(f"Memory mount '{mount_name}' is not attached to node '{node_manifest.node_id}'.")

    def _check_mount_access(
        self,
        node_manifest: NodeManifest,
        mount_name: str,
        action: str,
        *,
        task_policy: TaskPolicy | None = None,
    ):
        decision = memory_mount_allowed(mount_name, task_policy=task_policy)
        if not decision.allowed:
            raise PermissionError(decision.reason)
        decision = check_access(
            node_manifest,
            ResourceRequirement(
                resource=f"memory.{action}:{mount_name}",
                action=action,
                scope=mount_name,
            ),
        )
        if not decision.allowed:
            raise PermissionError(decision.reason)

    def private_buffer(
        self,
        node_manifest: NodeManifest,
        mount_name: str,
        *,
        task_policy: TaskPolicy | None = None,
    ) -> BufferMemory:
        mount = self._mount_for(node_manifest, mount_name)
        if mount.backend != "private_buffer":
            raise ValueError(f"Mount '{mount_name}' is not a private_buffer mount.")
        if "r" not in mount.mode:
            raise PermissionError(f"Mount '{mount_name}' is not readable.")
        self._check_mount_access(node_manifest, mount_name, "read", task_policy=task_policy)
        key = (node_manifest.node_id, mount_name)
        if key not in self._private_buffers:
            self._private_buffers[key] = BufferMemory()
        return self._private_buffers[key]

    def remember_private(
        self,
        mount_name: str,
        query: str,
        *,
        node_manifest: NodeManifest,
        task_policy: TaskPolicy | None = None,
        k: int = 5,
    ) -> list[str]:
        buffer = self.private_buffer(
            node_manifest=node_manifest,
            mount_name=mount_name,
            task_policy=task_policy,
        )
        return buffer.remember(query, k=k)

    def memorize_private(
        self,
        mount_name: str,
        content: str,
        *,
        node_manifest: NodeManifest,
        task_policy: TaskPolicy | None = None,
        metadata: dict | None = None,
    ):
        mount = self._mount_for(node_manifest, mount_name)
        if mount.backend != "private_buffer":
            raise ValueError(f"Mount '{mount_name}' is not a private_buffer mount.")
        if "w" not in mount.mode:
            raise PermissionError(f"Mount '{mount_name}' is not writable.")
        self._check_mount_access(node_manifest, mount_name, "write", task_policy=task_policy)
        buffer = self.private_buffer(
            node_manifest=node_manifest,
            mount_name=mount_name,
            task_policy=task_policy,
        )
        buffer.memorize(content, metadata=metadata)
        self._emit(
            "memory_written",
            node_id=node_manifest.node_id,
            message="private buffer written",
            metadata={"mount": mount_name},
        )

    def append_note(
        self,
        task_id: str,
        mount_name: str,
        note: str,
        *,
        node_manifest: NodeManifest,
        task_policy: TaskPolicy | None = None,
        metadata: dict | None = None,
    ):
        mount = self._mount_for(node_manifest, mount_name)
        if mount.backend != "task_scratchpad":
            raise ValueError(f"Mount '{mount_name}' does not support scratchpad notes.")
        if "w" not in mount.mode:
            raise PermissionError(f"Mount '{mount_name}' is not writable.")
        self._check_mount_access(node_manifest, mount_name, "write", task_policy=task_policy)
        self.shared_store.append_note(task_id, mount_name, note, node_manifest.node_id, metadata)
        self._emit("memory_written", task_id=task_id, node_id=node_manifest.node_id, message="note appended", metadata={"mount": mount_name})

    def list_notes(
        self,
        task_id: str,
        mount_name: str,
        *,
        node_manifest: NodeManifest,
        task_policy: TaskPolicy | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        mount = self._mount_for(node_manifest, mount_name)
        if mount.backend != "task_scratchpad":
            raise ValueError(f"Mount '{mount_name}' does not support scratchpad notes.")
        self._check_mount_access(node_manifest, mount_name, "read", task_policy=task_policy)
        return self.shared_store.list_notes(task_id, mount_name, limit=limit)

    def write_state(
        self,
        task_id: str,
        mount_name: str,
        state: dict[str, Any],
        *,
        node_manifest: NodeManifest,
        task_policy: TaskPolicy | None = None,
    ):
        mount = self._mount_for(node_manifest, mount_name)
        if mount.backend != "task_scratchpad":
            raise ValueError(f"Mount '{mount_name}' does not support task state.")
        if "w" not in mount.mode:
            raise PermissionError(f"Mount '{mount_name}' is not writable.")
        self._check_mount_access(node_manifest, mount_name, "write", task_policy=task_policy)

        if self.task_store is not None:
            record = self.task_store.get(task_id)
            if record is None:
                raise KeyError(f"Task '{task_id}' not found in task store.")
            if record.lease_owner != node_manifest.node_id:
                raise PermissionError("Only the current lease owner can write task scratchpad state.")

        self.shared_store.write_state(task_id, mount_name, state, node_manifest.node_id)
        self._emit("memory_written", task_id=task_id, node_id=node_manifest.node_id, message="state written", metadata={"mount": mount_name})

    def read_state(
        self,
        task_id: str,
        mount_name: str,
        *,
        node_manifest: NodeManifest,
        task_policy: TaskPolicy | None = None,
    ) -> dict[str, Any]:
        mount = self._mount_for(node_manifest, mount_name)
        if mount.backend != "task_scratchpad":
            raise ValueError(f"Mount '{mount_name}' does not support task state.")
        self._check_mount_access(node_manifest, mount_name, "read", task_policy=task_policy)
        return self.shared_store.read_state(task_id, mount_name)

    def semantic_memory(self, tenant_id: str, memory_scope: str):
        key = (tenant_id or "default", memory_scope or "default")
        if key not in self._semantic_memories:
            self._semantic_memories[key] = self.semantic_factory(key[0], key[1])
        return self._semantic_memories[key]

    def remember(
        self,
        mount_name: str,
        query: str,
        *,
        node_manifest: NodeManifest,
        tenant_id: str,
        memory_scope: str,
        task_policy: TaskPolicy | None = None,
        k: int = 5,
    ) -> list[str]:
        mount = self._mount_for(node_manifest, mount_name)
        if mount.backend != "semantic_shared":
            raise ValueError(f"Mount '{mount_name}' is not semantic_shared.")
        self._check_mount_access(node_manifest, mount_name, "read", task_policy=task_policy)
        return self.semantic_memory(tenant_id, memory_scope).remember(query, k=k)

    def memorize(
        self,
        mount_name: str,
        content: str,
        *,
        node_manifest: NodeManifest,
        tenant_id: str,
        memory_scope: str,
        task_policy: TaskPolicy | None = None,
        metadata: dict | None = None,
    ):
        mount = self._mount_for(node_manifest, mount_name)
        if mount.backend != "semantic_shared":
            raise ValueError(f"Mount '{mount_name}' is not semantic_shared.")
        if "w" not in mount.mode:
            raise PermissionError(f"Mount '{mount_name}' is not writable.")
        self._check_mount_access(node_manifest, mount_name, "write", task_policy=task_policy)
        self.semantic_memory(tenant_id, memory_scope).memorize(content, metadata=metadata)
        self._emit("memory_written", tenant_id=tenant_id, node_id=node_manifest.node_id, message="semantic memory written", metadata={"mount": mount_name, "memory_scope": memory_scope})
