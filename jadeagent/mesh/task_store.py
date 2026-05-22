"""
Durable task lifecycle storage for mesh execution.
"""

from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .audit import AuditEvent, coerce_audit_event
from .lease_wheel import LeaseDeadlineIndex
from .protocol import MeshTask, TaskResult, TaskState


@dataclass
class TaskRecord:
    task: MeshTask
    state: TaskState = TaskState.PENDING
    attempts: int = 0
    lease_owner: str = ""
    lease_deadline: float = 0.0
    result: TaskResult | None = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def task_id(self) -> str:
        return self.task.task_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": task_to_dict(self.task),
            "state": self.state.value,
            "attempts": self.attempts,
            "lease_owner": self.lease_owner,
            "lease_deadline": self.lease_deadline,
            "result": task_result_to_dict(self.result) if self.result else None,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskRecord:
        return cls(
            task=task_from_dict(dict(data.get("task", {}))),
            state=TaskState(str(data.get("state", TaskState.PENDING.value))),
            attempts=int(data.get("attempts", 0)),
            lease_owner=str(data.get("lease_owner", "")),
            lease_deadline=float(data.get("lease_deadline", 0.0)),
            result=task_result_from_dict(data["result"]) if data.get("result") else None,
            error=str(data.get("error", "")),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
        )

    def to_result(self) -> TaskResult:
        if self.result is not None:
            return self.result
        return TaskResult(
            task_id=self.task.task_id,
            capability=self.task.capability,
            node_id=self.lease_owner or self.task.requester,
            state=self.state,
            error=self.error or None,
            started_at=self.created_at,
            finished_at=self.updated_at if self.state in {
                TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED,
            } else None,
            metadata={
                "attempts": self.attempts,
                "tenant_id": self.task.tenant_id,
                "memory_scope": self.task.memory_scope,
            },
        )


def task_to_dict(task: MeshTask) -> dict[str, Any]:
    return {
        "capability": task.capability,
        "prompt": task.prompt,
        "requester": task.requester,
        "task_id": task.task_id,
        "priority": task.priority,
        "ttl": task.ttl,
        "affinity": task.affinity,
        "metadata": dict(task.metadata),
        "task_policy": dict(task.task_policy),
        "max_attempts": task.max_attempts,
        "lease_seconds": task.lease_seconds,
        "tenant_id": task.tenant_id,
        "memory_scope": task.memory_scope,
        "parent_task_id": task.parent_task_id,
        "min_trust_tier": task.min_trust_tier,
    }


def task_from_dict(data: dict[str, Any]) -> MeshTask:
    return MeshTask(
        capability=str(data.get("capability", "")),
        prompt=str(data.get("prompt", "")),
        requester=str(data.get("requester", "client")),
        task_id=str(data.get("task_id", "")),
        priority=int(data.get("priority", 0)),
        ttl=int(data.get("ttl", 8)),
        affinity=data.get("affinity"),
        metadata=dict(data.get("metadata", {})),
        task_policy=dict(data.get("task_policy", {})),
        max_attempts=int(data.get("max_attempts", 3)),
        lease_seconds=float(data.get("lease_seconds", 30.0)),
        tenant_id=str(data.get("tenant_id", "")),
        memory_scope=str(data.get("memory_scope", "")),
        parent_task_id=str(data.get("parent_task_id", "")) or None,
        min_trust_tier=str(data.get("min_trust_tier", "standard")),
    )


def task_result_to_dict(result: TaskResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "task_id": result.task_id,
        "capability": result.capability,
        "node_id": result.node_id,
        "state": result.state.value,
        "output": result.output,
        "error": result.error,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "metadata": dict(result.metadata),
    }


def task_result_from_dict(data: dict[str, Any]) -> TaskResult:
    return TaskResult(
        task_id=str(data.get("task_id", "")),
        capability=str(data.get("capability", "")),
        node_id=str(data.get("node_id", "")),
        state=TaskState(str(data.get("state", TaskState.FAILED.value))),
        output=str(data.get("output", "")),
        error=data.get("error"),
        started_at=float(data.get("started_at", time.time())),
        finished_at=float(data["finished_at"]) if data.get("finished_at") is not None else None,
        metadata=dict(data.get("metadata", {})),
    )


class TaskStore(ABC):
    @abstractmethod
    def submit(self, task: MeshTask) -> TaskRecord:
        ...

    @abstractmethod
    def claim_next(self, node_id: str, capability: str) -> TaskRecord | None:
        ...

    @abstractmethod
    def renew_lease(self, task_id: str, node_id: str, lease_seconds: float | None = None) -> TaskRecord | None:
        ...

    @abstractmethod
    def complete(self, task_id: str, node_id: str, result: TaskResult) -> TaskRecord | None:
        ...

    @abstractmethod
    def fail(self, task_id: str, node_id: str, error: str) -> TaskRecord | None:
        ...

    @abstractmethod
    def cancel(self, task_id: str, reason: str = "cancelled") -> TaskRecord | None:
        ...

    @abstractmethod
    def requeue_expired(self) -> int:
        ...

    @abstractmethod
    def get(self, task_id: str) -> TaskRecord | None:
        ...

    @abstractmethod
    def list_events(self, task_id: str | None = None, limit: int = 100) -> list[AuditEvent]:
        ...

    @abstractmethod
    def record_event(self, event: AuditEvent | dict[str, Any]):
        ...


class InMemoryTaskStore(TaskStore):
    def __init__(self):
        self._tasks: dict[str, TaskRecord] = {}
        self._queues: dict[str, list[tuple[float, float, int, str]]] = {}
        self._leases = LeaseDeadlineIndex()
        self._events: list[AuditEvent] = []
        self._counter = 0
        self._lock = threading.RLock()

    def _enqueue(self, record: TaskRecord):
        self._counter += 1
        queue = self._queues.setdefault(record.task.capability, [])
        item = (-float(record.task.priority), record.created_at, self._counter, record.task_id)
        queue.append(item)
        queue.sort()

    def _emit(self, event_type: str, record: TaskRecord, message: str = "", metadata: dict[str, Any] | None = None):
        self.record_event(AuditEvent(
            event_type=event_type,
            task_id=record.task.task_id,
            node_id=record.lease_owner,
            tenant_id=record.task.tenant_id,
            parent_task_id=record.task.parent_task_id or "",
            message=message,
            metadata=dict(metadata or {}),
        ))

    def submit(self, task: MeshTask) -> TaskRecord:
        with self._lock:
            now = time.time()
            record = TaskRecord(task=task, created_at=now, updated_at=now)
            self._tasks[task.task_id] = record
            self._enqueue(record)
            self._emit("task_submitted", record, "task submitted")
            return record

    def claim_next(self, node_id: str, capability: str) -> TaskRecord | None:
        with self._lock:
            self.requeue_expired()
            queue = self._queues.setdefault(capability, [])
            while queue:
                _, _, _, task_id = queue.pop(0)
                record = self._tasks.get(task_id)
                if record is None or record.state != TaskState.PENDING:
                    continue
                now = time.time()
                record.state = TaskState.RUNNING
                record.attempts += 1
                record.lease_owner = node_id
                record.lease_deadline = now + max(record.task.lease_seconds, 0.1)
                record.updated_at = now
                self._leases.upsert(record.task_id, node_id, record.lease_deadline)
                self._emit("task_claimed", record, "task claimed")
                return record
            return None

    def renew_lease(self, task_id: str, node_id: str, lease_seconds: float | None = None) -> TaskRecord | None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None or record.state != TaskState.RUNNING or record.lease_owner != node_id:
                return None
            record.lease_deadline = time.time() + max(lease_seconds or record.task.lease_seconds, 0.1)
            record.updated_at = time.time()
            self._leases.upsert(task_id, node_id, record.lease_deadline)
            return record

    def complete(self, task_id: str, node_id: str, result: TaskResult) -> TaskRecord | None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None
            if record.state == TaskState.COMPLETED:
                return record
            record.state = TaskState.COMPLETED
            record.result = result
            record.lease_owner = node_id
            record.lease_deadline = 0.0
            record.error = ""
            record.updated_at = time.time()
            self._leases.discard(task_id)
            self._emit("task_completed", record, "task completed")
            return record

    def fail(self, task_id: str, node_id: str, error: str) -> TaskRecord | None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None
            record.updated_at = time.time()
            record.error = error
            retrying = record.attempts < max(record.task.max_attempts, 1)
            self._leases.discard(task_id)
            if retrying:
                record.state = TaskState.PENDING
                record.lease_owner = ""
                record.lease_deadline = 0.0
                self._enqueue(record)
            else:
                record.state = TaskState.FAILED
                record.lease_owner = node_id
                record.lease_deadline = 0.0
            self._emit("task_failed", record, error, {"retrying": retrying})
            return record

    def cancel(self, task_id: str, reason: str = "cancelled") -> TaskRecord | None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None
            if record.state in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}:
                return record
            record.state = TaskState.CANCELLED
            record.error = reason
            record.lease_owner = ""
            record.lease_deadline = 0.0
            record.updated_at = time.time()
            self._leases.discard(task_id)
            self._emit("task_failed", record, reason, {"cancelled": True})
            return record

    def requeue_expired(self) -> int:
        with self._lock:
            now = time.time()
            count = 0
            for lease in self._leases.pop_expired(now=now):
                record = self._tasks.get(lease.task_id)
                if record is None:
                    continue
                if record.state != TaskState.RUNNING or record.lease_owner != lease.owner:
                    continue
                count += 1
                self._emit("lease_expired", record, "lease expired")
                if record.attempts < max(record.task.max_attempts, 1):
                    record.state = TaskState.PENDING
                    record.lease_owner = ""
                    record.lease_deadline = 0.0
                    record.updated_at = now
                    self._enqueue(record)
                else:
                    record.state = TaskState.FAILED
                    record.error = record.error or "lease expired and retry budget exhausted"
                    record.lease_deadline = 0.0
                    record.updated_at = now
                    self._emit("task_failed", record, record.error, {"retrying": False})
            return count

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)

    def record_event(self, event: AuditEvent | dict[str, Any]):
        with self._lock:
            self._events.append(coerce_audit_event(event))

    def list_events(self, task_id: str | None = None, limit: int = 100) -> list[AuditEvent]:
        with self._lock:
            events = self._events
            if task_id:
                events = [event for event in events if event.task_id == task_id]
            return list(events[-limit:])


class RedisTaskStore(TaskStore):
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "jade:taskstore",
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
                "RedisTaskStore requires redis package. Install with: pip install redis"
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

    def _task_key(self, task_id: str) -> str:
        return f"{self.key_prefix}:task:{task_id}"

    def _queue_key(self, capability: str) -> str:
        return f"{self.key_prefix}:queue:{capability}"

    def _worker_key(self, node_id: str) -> str:
        return f"{self.key_prefix}:worker:{node_id}"

    def _events_key(self) -> str:
        return f"{self.key_prefix}:events"

    def _leases_key(self) -> str:
        return f"{self.key_prefix}:leases"

    def _score(self, task: MeshTask) -> float:
        return time.time() - (float(task.priority) * 1_000_000.0)

    def submit(self, task: MeshTask) -> TaskRecord:
        now = time.time()
        record = TaskRecord(task=task, created_at=now, updated_at=now)
        pipe = self._client.pipeline(transaction=False)
        pipe.hset(self._task_key(task.task_id), mapping={
            "record": json.dumps(record.to_dict(), separators=(",", ":")),
        })
        pipe.zadd(self._queue_key(task.capability), {task.task_id: self._score(task)})
        pipe.execute()
        self.record_event(AuditEvent(
            event_type="task_submitted",
            task_id=task.task_id,
            tenant_id=task.tenant_id,
            parent_task_id=task.parent_task_id or "",
            message="task submitted",
        ))
        return record

    def claim_next(self, node_id: str, capability: str) -> TaskRecord | None:
        self.requeue_expired()
        queue_key = self._queue_key(capability)
        while True:
            candidates = self._client.zrange(queue_key, 0, 0)
            if not candidates:
                return None
            task_id = candidates[0]
            task_key = self._task_key(task_id)
            with self._client.pipeline() as pipe:
                try:
                    pipe.watch(task_key, queue_key)
                    raw = pipe.hget(task_key, "record")
                    if not raw:
                        pipe.multi()
                        pipe.zrem(queue_key, task_id)
                        pipe.execute()
                        continue
                    record = TaskRecord.from_dict(json.loads(raw))
                    if record.state != TaskState.PENDING:
                        pipe.multi()
                        pipe.zrem(queue_key, task_id)
                        pipe.execute()
                        continue
                    now = time.time()
                    record.state = TaskState.RUNNING
                    record.attempts += 1
                    record.lease_owner = node_id
                    record.lease_deadline = now + max(record.task.lease_seconds, 0.1)
                    record.updated_at = now
                    pipe.multi()
                    pipe.hset(task_key, "record", json.dumps(record.to_dict(), separators=(",", ":")))
                    pipe.zrem(queue_key, task_id)
                    pipe.sadd(self._worker_key(node_id), task_id)
                    pipe.zadd(self._leases_key(), {task_id: record.lease_deadline})
                    pipe.execute()
                    self.record_event(AuditEvent(
                        event_type="task_claimed",
                        task_id=task_id,
                        node_id=node_id,
                        tenant_id=record.task.tenant_id,
                        parent_task_id=record.task.parent_task_id or "",
                        message="task claimed",
                    ))
                    return record
                except Exception:
                    continue

    def renew_lease(self, task_id: str, node_id: str, lease_seconds: float | None = None) -> TaskRecord | None:
        record = self.get(task_id)
        if record is None or record.state != TaskState.RUNNING or record.lease_owner != node_id:
            return None
        record.lease_deadline = time.time() + max(lease_seconds or record.task.lease_seconds, 0.1)
        record.updated_at = time.time()
        pipe = self._client.pipeline(transaction=False)
        pipe.hset(self._task_key(task_id), "record", json.dumps(record.to_dict(), separators=(",", ":")))
        pipe.zadd(self._leases_key(), {task_id: record.lease_deadline})
        pipe.execute()
        return record

    def complete(self, task_id: str, node_id: str, result: TaskResult) -> TaskRecord | None:
        record = self.get(task_id)
        if record is None:
            return None
        record.state = TaskState.COMPLETED
        record.result = result
        record.lease_owner = node_id
        record.lease_deadline = 0.0
        record.error = ""
        record.updated_at = time.time()
        pipe = self._client.pipeline(transaction=False)
        pipe.hset(self._task_key(task_id), "record", json.dumps(record.to_dict(), separators=(",", ":")))
        pipe.srem(self._worker_key(node_id), task_id)
        pipe.zrem(self._leases_key(), task_id)
        pipe.execute()
        self.record_event(AuditEvent(
            event_type="task_completed",
            task_id=task_id,
            node_id=node_id,
            tenant_id=record.task.tenant_id,
            parent_task_id=record.task.parent_task_id or "",
            message="task completed",
        ))
        return record

    def fail(self, task_id: str, node_id: str, error: str) -> TaskRecord | None:
        record = self.get(task_id)
        if record is None:
            return None
        retrying = record.attempts < max(record.task.max_attempts, 1)
        record.updated_at = time.time()
        record.error = error
        pipe = self._client.pipeline(transaction=False)
        pipe.zrem(self._leases_key(), task_id)
        if retrying:
            record.state = TaskState.PENDING
            record.lease_owner = ""
            record.lease_deadline = 0.0
            pipe.zadd(self._queue_key(record.task.capability), {task_id: self._score(record.task)})
        else:
            record.state = TaskState.FAILED
            record.lease_owner = node_id
            record.lease_deadline = 0.0
        pipe.hset(self._task_key(task_id), "record", json.dumps(record.to_dict(), separators=(",", ":")))
        pipe.srem(self._worker_key(node_id), task_id)
        pipe.execute()
        self.record_event(AuditEvent(
            event_type="task_failed",
            task_id=task_id,
            node_id=node_id,
            tenant_id=record.task.tenant_id,
            parent_task_id=record.task.parent_task_id or "",
            message=error,
            metadata={"retrying": retrying},
        ))
        return record

    def cancel(self, task_id: str, reason: str = "cancelled") -> TaskRecord | None:
        record = self.get(task_id)
        if record is None:
            return None
        if record.state in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}:
            return record
        previous_owner = record.lease_owner
        record.state = TaskState.CANCELLED
        record.error = reason
        record.lease_owner = ""
        record.lease_deadline = 0.0
        record.updated_at = time.time()
        pipe = self._client.pipeline(transaction=False)
        pipe.hset(self._task_key(task_id), "record", json.dumps(record.to_dict(), separators=(",", ":")))
        pipe.zrem(self._leases_key(), task_id)
        if previous_owner:
            pipe.srem(self._worker_key(previous_owner), task_id)
        pipe.execute()
        self.record_event(AuditEvent(
            event_type="task_failed",
            task_id=task_id,
            tenant_id=record.task.tenant_id,
            parent_task_id=record.task.parent_task_id or "",
            message=reason,
            metadata={"cancelled": True},
        ))
        return record

    def requeue_expired(self) -> int:
        count = 0
        now = time.time()
        task_ids = self._client.zrangebyscore(self._leases_key(), 0, now)
        for task_id in task_ids:
            record = self.get(task_id)
            if record is None:
                self._client.zrem(self._leases_key(), task_id)
                continue
            if record.state != TaskState.RUNNING or record.lease_deadline <= 0 or record.lease_deadline > now:
                continue
            count += 1
            self.record_event(AuditEvent(
                event_type="lease_expired",
                task_id=task_id,
                node_id=record.lease_owner,
                tenant_id=record.task.tenant_id,
                parent_task_id=record.task.parent_task_id or "",
                message="lease expired",
            ))
            self.fail(task_id, record.lease_owner, record.error or "lease expired")
        return count

    def get(self, task_id: str) -> TaskRecord | None:
        raw = self._client.hget(self._task_key(task_id), "record")
        if not raw:
            return None
        return TaskRecord.from_dict(json.loads(raw))

    def record_event(self, event: AuditEvent | dict[str, Any]):
        audit_event = coerce_audit_event(event)
        payload = audit_event.to_dict()
        payload["metadata"] = json.dumps(payload.get("metadata", {}), separators=(",", ":"))
        self._client.xadd(self._events_key(), payload)

    def list_events(self, task_id: str | None = None, limit: int = 100) -> list[AuditEvent]:
        rows = self._client.xrevrange(self._events_key(), count=max(limit, 1))
        events = [AuditEvent.from_dict(row) for _, row in reversed(rows)]
        if task_id:
            events = [event for event in events if event.task_id == task_id]
        return events[-limit:]
