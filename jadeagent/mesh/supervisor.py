"""
Local shard supervisor with ready and retry queues.
"""

from __future__ import annotations

import asyncio
import heapq
import time
from collections import defaultdict
from typing import Any

from .async_node import AsyncMeshNode
from .protocol import MeshTask, TaskResult
from .worker_pool import LocalWorkerIndex, WorkerState


class ShardSupervisor:
    """
    Local control-plane owner for one `(tenant_id, capability)` shard.
    """

    def __init__(
        self,
        supervisor_id: str,
        *,
        tenant_id: str,
        capability: str,
        retry_delay_seconds: float = 0.0,
        audit_sink: Any = None,
    ):
        self.supervisor_id = supervisor_id
        self.tenant_id = tenant_id
        self.capability = capability
        self.retry_delay_seconds = float(retry_delay_seconds)
        self.audit_sink = audit_sink

        self._ready: list[tuple[int, int, MeshTask]] = []
        self._retry: list[tuple[float, int, MeshTask]] = []
        self._dead_letter: dict[str, TaskResult] = {}
        self._completed: dict[str, TaskResult] = {}
        self._attempts: dict[str, int] = {}
        self._worker_index = LocalWorkerIndex()
        self._stats: dict[str, int] = defaultdict(int)
        self._last_updated_at = time.time()
        self._seq = 0
        self._lock = asyncio.Lock()

    @property
    def dead_letter(self) -> dict[str, TaskResult]:
        return self._dead_letter

    @property
    def completed(self) -> dict[str, TaskResult]:
        return self._completed

    @property
    def queue_depth(self) -> int:
        return len(self._ready) + len(self._retry)

    def matches(self, task: MeshTask) -> bool:
        return task.capability == self.capability and task.tenant_id == self.tenant_id

    def register_worker(
        self,
        worker: AsyncMeshNode,
        *,
        permits: int = 1,
        metadata: dict[str, Any] | None = None,
    ):
        return self._worker_index.register_worker(
            worker=worker,
            permits=max(int(permits), 1),
            metadata=dict(metadata or {}),
        )

    def unregister_worker(self, node_id: str):
        self._worker_index.unregister_worker(node_id)

    def update_worker(
        self,
        node_id: str,
        *,
        permits: int | None = None,
        inflight: int | None = None,
        metadata: dict[str, Any] | None = None,
        merge_metadata: bool = True,
    ) -> WorkerState | None:
        return self._worker_index.update_worker(
            node_id,
            permits=permits,
            inflight=inflight,
            metadata=metadata,
            merge_metadata=merge_metadata,
        )

    async def submit(self, task: MeshTask):
        if not self.matches(task):
            raise ValueError(
                f"Task '{task.task_id}' does not belong to shard ({self.tenant_id}, {self.capability})."
            )
        async with self._lock:
            self._seq += 1
            heapq.heappush(self._ready, (-int(task.priority), self._seq, task))
        await self._emit("task_submitted", task, "task submitted to shard supervisor")

    async def _emit(self, event_type: str, task: MeshTask, message: str, metadata: dict[str, Any] | None = None):
        self._stats[event_type] += 1
        self._last_updated_at = time.time()
        if self.audit_sink is None:
            return
        record_fn = getattr(self.audit_sink, "record_event", None)
        if not callable(record_fn):
            return
        payload = {
            "event_type": event_type,
            "task_id": task.task_id,
            "tenant_id": task.tenant_id,
            "parent_task_id": task.parent_task_id or "",
            "node_id": self.supervisor_id,
            "message": message,
            "metadata": {
                "capability": task.capability,
                **dict(metadata or {}),
            },
        }
        result = record_fn(payload)
        if asyncio.iscoroutine(result):
            await result

    async def _promote_due_retries(self):
        async with self._lock:
            now = time.time()
            while self._retry and self._retry[0][0] <= now:
                _, _, task = heapq.heappop(self._retry)
                self._seq += 1
                heapq.heappush(self._ready, (-int(task.priority), self._seq, task))

    def _select_worker(self, task: MeshTask) -> WorkerState | None:
        return self._worker_index.select_worker(task, backlog_depth=self.queue_depth)

    async def dispatch_once(self) -> bool:
        await self._promote_due_retries()

        async with self._lock:
            if not self._ready:
                return False
            _, _, task = heapq.heappop(self._ready)

        worker_state = self._select_worker(task)
        if worker_state is None:
            async with self._lock:
                self._seq += 1
                heapq.heappush(self._ready, (-int(task.priority), self._seq, task))
            return False

        if not self._worker_index.reserve(worker_state.worker.node_id):
            async with self._lock:
                self._seq += 1
                heapq.heappush(self._ready, (-int(task.priority), self._seq, task))
            return False
        self._attempts[task.task_id] = self._attempts.get(task.task_id, 0) + 1
        await self._emit("task_claimed", task, "task claimed by shard supervisor", {"worker_id": worker_state.worker.node_id})
        try:
            result = await worker_state.worker.execute_assigned_task(task)
        finally:
            self._worker_index.release(worker_state.worker.node_id)

        if result.success:
            self._completed[result.task_id] = result
            await self._emit("task_completed", task, "task completed", {"worker_id": result.node_id})
            return True

        attempts = self._attempts.get(task.task_id, 1)
        if attempts < max(int(task.max_attempts), 1):
            ready_at = time.time() + max(self.retry_delay_seconds, 0.0)
            async with self._lock:
                self._seq += 1
                heapq.heappush(self._retry, (ready_at, self._seq, task))
            await self._emit(
                "task_failed",
                task,
                result.error or "task failed",
                {"worker_id": result.node_id, "retrying": True, "attempts": attempts},
            )
        else:
            self._dead_letter[result.task_id] = result
            await self._emit(
                "task_failed",
                task,
                result.error or "task failed",
                {"worker_id": result.node_id, "retrying": False, "attempts": attempts},
            )
        return True

    async def run_until_idle(self, max_cycles: int = 1000) -> int:
        cycles = 0
        for _ in range(max_cycles):
            progressed = await self.dispatch_once()
            cycles += 1
            if progressed:
                continue

            await self._promote_due_retries()
            async with self._lock:
                if self._ready:
                    continue
                if not self._retry:
                    break
                next_ready_at = self._retry[0][0]

            sleep_for = max(next_ready_at - time.time(), 0.0)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        return cycles

    def snapshot(self) -> dict[str, Any]:
        worker_metrics = self._worker_index.aggregate()
        return {
            "supervisor_id": self.supervisor_id,
            "shard_id": f"{self.tenant_id or 'default'}::{self.capability}",
            "tenant_id": self.tenant_id,
            "capability": self.capability,
            "ready_depth": len(self._ready),
            "retry_depth": len(self._retry),
            "dead_letter": len(self._dead_letter),
            "completed": len(self._completed),
            "stats": dict(sorted(self._stats.items())),
            "last_updated_at": self._last_updated_at,
            "worker_metrics": worker_metrics,
            "workers": self._worker_index.snapshot(),
        }
