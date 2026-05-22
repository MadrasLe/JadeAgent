"""
Deadline-aware lease tracking primitives.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class LeaseRecord:
    task_id: str
    owner: str
    deadline: float
    version: int


class LeaseDeadlineIndex:
    """
    Min-heap lease index with stale-entry eviction.

    This gives the runtime an O(log N) path for lease scheduling and expiry
    checks, instead of broad scans over every running task.
    """

    def __init__(self):
        self._heap: list[tuple[float, int, str]] = []
        self._active: dict[str, LeaseRecord] = {}

    def __len__(self) -> int:
        return len(self._active)

    def upsert(self, task_id: str, owner: str, deadline: float) -> LeaseRecord:
        current = self._active.get(task_id)
        version = 1 if current is None else current.version + 1
        record = LeaseRecord(
            task_id=str(task_id),
            owner=str(owner),
            deadline=float(deadline),
            version=version,
        )
        self._active[task_id] = record
        heapq.heappush(self._heap, (record.deadline, record.version, record.task_id))
        return record

    def discard(self, task_id: str) -> LeaseRecord | None:
        return self._active.pop(task_id, None)

    def get(self, task_id: str) -> LeaseRecord | None:
        return self._active.get(task_id)

    def next_deadline(self) -> float | None:
        self._prune_stale()
        if not self._heap:
            return None
        return float(self._heap[0][0])

    def pop_expired(self, now: float | None = None, limit: int | None = None) -> list[LeaseRecord]:
        now_ts = time.time() if now is None else float(now)
        expired: list[LeaseRecord] = []

        self._prune_stale()
        while self._heap and self._heap[0][0] <= now_ts:
            deadline, version, task_id = heapq.heappop(self._heap)
            current = self._active.get(task_id)
            if current is None or current.version != version or current.deadline != deadline:
                continue
            self._active.pop(task_id, None)
            expired.append(current)
            if limit is not None and len(expired) >= max(int(limit), 0):
                break

        return expired

    def snapshot(self) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for record in sorted(self._active.values(), key=lambda item: (item.deadline, item.task_id)):
            rows.append({
                "task_id": record.task_id,
                "owner": record.owner,
                "deadline": record.deadline,
                "version": record.version,
            })
        return rows

    def _prune_stale(self):
        while self._heap:
            deadline, version, task_id = self._heap[0]
            current = self._active.get(task_id)
            if current is None or current.version != version or current.deadline != deadline:
                heapq.heappop(self._heap)
                continue
            break
