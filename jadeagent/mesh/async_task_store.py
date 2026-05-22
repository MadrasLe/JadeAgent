"""
Async task lifecycle storage for mesh execution.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from .audit import AuditEvent, coerce_audit_event
from .lease_wheel import LeaseDeadlineIndex
from .protocol import MeshTask, TaskResult, TaskState
from .task_store import InMemoryTaskStore, TaskRecord, TaskStore


class AsyncTaskStore(ABC):
    @abstractmethod
    async def submit(self, task: MeshTask) -> TaskRecord:
        ...

    @abstractmethod
    async def claim_next(self, node_id: str, capability: str) -> TaskRecord | None:
        ...

    @abstractmethod
    async def renew_lease(
        self,
        task_id: str,
        node_id: str,
        lease_seconds: float | None = None,
    ) -> TaskRecord | None:
        ...

    @abstractmethod
    async def complete(self, task_id: str, node_id: str, result: TaskResult) -> TaskRecord | None:
        ...

    @abstractmethod
    async def fail(self, task_id: str, node_id: str, error: str) -> TaskRecord | None:
        ...

    @abstractmethod
    async def cancel(self, task_id: str, reason: str = "cancelled") -> TaskRecord | None:
        ...

    @abstractmethod
    async def requeue_expired(self) -> int:
        ...

    @abstractmethod
    async def get(self, task_id: str) -> TaskRecord | None:
        ...

    @abstractmethod
    async def list_events(self, task_id: str | None = None, limit: int = 100) -> list[AuditEvent]:
        ...

    @abstractmethod
    async def record_event(self, event: AuditEvent | dict[str, Any]):
        ...

    @abstractmethod
    async def wait_for_terminal(
        self,
        task_id: str,
        timeout: float | None = None,
    ) -> TaskRecord | None:
        ...


class AsyncTaskStoreAdapter(AsyncTaskStore):
    """
    Async adapter around an existing sync TaskStore.

    This preserves compatibility while native async stores are adopted. It does
    not provide the same hot-path guarantees as a native event-driven store.
    """

    def __init__(self, store: TaskStore):
        self.store = store

    async def submit(self, task: MeshTask) -> TaskRecord:
        return await asyncio.to_thread(self.store.submit, task)

    async def claim_next(self, node_id: str, capability: str) -> TaskRecord | None:
        return await asyncio.to_thread(self.store.claim_next, node_id, capability)

    async def renew_lease(
        self,
        task_id: str,
        node_id: str,
        lease_seconds: float | None = None,
    ) -> TaskRecord | None:
        return await asyncio.to_thread(self.store.renew_lease, task_id, node_id, lease_seconds)

    async def complete(self, task_id: str, node_id: str, result: TaskResult) -> TaskRecord | None:
        return await asyncio.to_thread(self.store.complete, task_id, node_id, result)

    async def fail(self, task_id: str, node_id: str, error: str) -> TaskRecord | None:
        return await asyncio.to_thread(self.store.fail, task_id, node_id, error)

    async def cancel(self, task_id: str, reason: str = "cancelled") -> TaskRecord | None:
        return await asyncio.to_thread(self.store.cancel, task_id, reason)

    async def requeue_expired(self) -> int:
        return int(await asyncio.to_thread(self.store.requeue_expired))

    async def get(self, task_id: str) -> TaskRecord | None:
        return await asyncio.to_thread(self.store.get, task_id)

    async def list_events(self, task_id: str | None = None, limit: int = 100) -> list[AuditEvent]:
        return await asyncio.to_thread(self.store.list_events, task_id, limit)

    async def record_event(self, event: AuditEvent | dict[str, Any]):
        await asyncio.to_thread(self.store.record_event, event)

    async def wait_for_terminal(
        self,
        task_id: str,
        timeout: float | None = None,
    ) -> TaskRecord | None:
        deadline = None if timeout is None else time.time() + max(float(timeout), 0.0)
        while True:
            record = await self.get(task_id)
            if record is not None and record.state in {
                TaskState.COMPLETED,
                TaskState.FAILED,
                TaskState.CANCELLED,
            }:
                return record
            if deadline is not None and time.time() >= deadline:
                return None
            await asyncio.sleep(0.05)


class AsyncInMemoryTaskStore(AsyncTaskStore):
    """
    Event-driven in-memory async task store.

    This store is designed for the async mesh runtime and avoids polling in the
    common wait paths by using asyncio events.
    """

    def __init__(self):
        self._tasks: dict[str, TaskRecord] = {}
        self._queues: dict[str, list[tuple[float, float, int, str]]] = defaultdict(list)
        self._leases = LeaseDeadlineIndex()
        self._events: list[AuditEvent] = []
        self._counter = 0
        self._lock = asyncio.Lock()
        self._ready_event = asyncio.Event()
        self._lease_event = asyncio.Event()
        self._terminal_events: dict[str, asyncio.Event] = {}

    def _terminal_event_for(self, task_id: str) -> asyncio.Event:
        event = self._terminal_events.get(task_id)
        if event is None:
            event = asyncio.Event()
            self._terminal_events[task_id] = event
        return event

    def _enqueue(self, record: TaskRecord):
        self._counter += 1
        item = (-float(record.task.priority), record.created_at, self._counter, record.task_id)
        queue = self._queues.setdefault(record.task.capability, [])
        queue.append(item)
        queue.sort()
        self._ready_event.set()

    async def _emit(
        self,
        event_type: str,
        record: TaskRecord,
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        await self.record_event(AuditEvent(
            event_type=event_type,
            task_id=record.task.task_id,
            node_id=record.lease_owner,
            tenant_id=record.task.tenant_id,
            parent_task_id=record.task.parent_task_id or "",
            message=message,
            metadata=dict(metadata or {}),
        ))

    async def submit(self, task: MeshTask) -> TaskRecord:
        async with self._lock:
            now = time.time()
            record = TaskRecord(task=task, created_at=now, updated_at=now)
            self._tasks[task.task_id] = record
            self._terminal_event_for(task.task_id)
            self._enqueue(record)
        await self._emit("task_submitted", record, "task submitted")
        return record

    async def claim_next(self, node_id: str, capability: str) -> TaskRecord | None:
        claimed: TaskRecord | None = None
        async with self._lock:
            await self._requeue_expired_locked()
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
                self._leases.upsert(task_id, node_id, record.lease_deadline)
                self._lease_event.set()
                if not any(self._queues.values()):
                    self._ready_event.clear()
                claimed = record
                break
            if not any(self._queues.values()):
                self._ready_event.clear()
        if claimed is not None:
            await self._emit("task_claimed", claimed, "task claimed")
        return claimed

    async def claim_next_available(
        self,
        node_id: str,
        capabilities: list[str] | tuple[str, ...] | set[str],
        timeout: float | None = None,
    ) -> TaskRecord | None:
        deadline = None if timeout is None else time.time() + max(float(timeout), 0.0)
        ordered = [str(capability) for capability in capabilities]

        while True:
            for capability in ordered:
                record = await self.claim_next(node_id, capability)
                if record is not None:
                    return record

            if deadline is not None and time.time() >= deadline:
                return None

            async with self._lock:
                next_lease_deadline = self._leases.next_deadline()

            now = time.time()
            wait_timeout = None
            if next_lease_deadline is not None:
                wait_timeout = max(next_lease_deadline - now, 0.0)
            if deadline is not None:
                remaining = max(deadline - now, 0.0)
                wait_timeout = remaining if wait_timeout is None else min(wait_timeout, remaining)
                if remaining <= 0:
                    return None

            if wait_timeout is not None and wait_timeout <= 0:
                await self.requeue_expired()
                continue

            ready_task = asyncio.create_task(self._ready_event.wait())
            lease_task = asyncio.create_task(self._lease_event.wait())
            done: set[asyncio.Task[Any]] = set()

            try:
                done, pending = await asyncio.wait(
                    {ready_task, lease_task},
                    timeout=wait_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                for pending_task in (ready_task, lease_task):
                    if pending_task not in done:
                        pending_task.cancel()
                await asyncio.gather(ready_task, lease_task, return_exceptions=True)

            if not done:
                await self.requeue_expired()
                continue

            if lease_task in done:
                async with self._lock:
                    self._lease_event.clear()
            async with self._lock:
                if not any(self._queues.values()):
                    self._ready_event.clear()

    async def renew_lease(
        self,
        task_id: str,
        node_id: str,
        lease_seconds: float | None = None,
    ) -> TaskRecord | None:
        async with self._lock:
            record = self._tasks.get(task_id)
            if record is None or record.state != TaskState.RUNNING or record.lease_owner != node_id:
                return None
            record.lease_deadline = time.time() + max(lease_seconds or record.task.lease_seconds, 0.1)
            record.updated_at = time.time()
            self._leases.upsert(task_id, node_id, record.lease_deadline)
            self._lease_event.set()
            return record

    async def complete(self, task_id: str, node_id: str, result: TaskResult) -> TaskRecord | None:
        async with self._lock:
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
            self._lease_event.set()
            self._terminal_event_for(task_id).set()
        await self._emit("task_completed", record, "task completed")
        return record

    async def fail(self, task_id: str, node_id: str, error: str) -> TaskRecord | None:
        async with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None
            record.updated_at = time.time()
            record.error = error
            retrying = record.attempts < max(record.task.max_attempts, 1)
            self._leases.discard(task_id)
            self._lease_event.set()
            if retrying:
                record.state = TaskState.PENDING
                record.lease_owner = ""
                record.lease_deadline = 0.0
                self._enqueue(record)
            else:
                record.state = TaskState.FAILED
                record.lease_owner = node_id
                record.lease_deadline = 0.0
                self._terminal_event_for(task_id).set()
        await self._emit("task_failed", record, error, {"retrying": retrying})
        return record

    async def cancel(self, task_id: str, reason: str = "cancelled") -> TaskRecord | None:
        async with self._lock:
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
            self._lease_event.set()
            self._terminal_event_for(task_id).set()
        await self._emit("task_failed", record, reason, {"cancelled": True})
        return record

    async def _requeue_expired_locked(self) -> int:
        now = time.time()
        count = 0
        for lease in self._leases.pop_expired(now=now):
            record = self._tasks.get(lease.task_id)
            if record is None:
                continue
            if record.state != TaskState.RUNNING or record.lease_owner != lease.owner:
                continue
            count += 1
            self._events.append(AuditEvent(
                event_type="lease_expired",
                task_id=record.task.task_id,
                node_id=record.lease_owner,
                tenant_id=record.task.tenant_id,
                parent_task_id=record.task.parent_task_id or "",
                message="lease expired",
            ))
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
                self._terminal_event_for(record.task.task_id).set()
                self._events.append(AuditEvent(
                    event_type="task_failed",
                    task_id=record.task.task_id,
                    node_id=record.lease_owner,
                    tenant_id=record.task.tenant_id,
                    parent_task_id=record.task.parent_task_id or "",
                    message=record.error,
                    metadata={"retrying": False},
                ))
        return count

    async def requeue_expired(self) -> int:
        async with self._lock:
            return await self._requeue_expired_locked()

    async def get(self, task_id: str) -> TaskRecord | None:
        async with self._lock:
            return self._tasks.get(task_id)

    async def record_event(self, event: AuditEvent | dict[str, Any]):
        async with self._lock:
            self._events.append(coerce_audit_event(event))

    async def list_events(self, task_id: str | None = None, limit: int = 100) -> list[AuditEvent]:
        async with self._lock:
            events = self._events
            if task_id:
                events = [event for event in events if event.task_id == task_id]
            return list(events[-limit:])

    async def wait_for_terminal(
        self,
        task_id: str,
        timeout: float | None = None,
    ) -> TaskRecord | None:
        record = await self.get(task_id)
        if record is not None and record.state in {
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELLED,
        }:
            return record

        async with self._lock:
            event = self._terminal_event_for(task_id)

        try:
            if timeout is None:
                await event.wait()
            else:
                await asyncio.wait_for(event.wait(), timeout=max(float(timeout), 0.0))
        except asyncio.TimeoutError:
            return None
        return await self.get(task_id)


def adapt_task_store(store: TaskStore | AsyncTaskStore) -> AsyncTaskStore:
    """Wrap a sync store when needed, preserving async call sites."""
    if isinstance(store, AsyncTaskStore):
        return store
    return AsyncTaskStoreAdapter(store)


def from_sync_in_memory(store: InMemoryTaskStore) -> AsyncTaskStoreAdapter:
    """Convenience wrapper for existing sync in-memory stores."""
    return AsyncTaskStoreAdapter(store)
