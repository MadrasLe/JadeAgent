"""
Local worker index and scoring primitives for shard scheduling.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from ..governance import TRUST_TIER_RANK
from .async_node import AsyncMeshNode
from .protocol import MeshTask


@dataclass
class WorkerState:
    worker: AsyncMeshNode
    permits: int = 1
    inflight: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.health > 0.0 and self.inflight < max(self.permits, 1)

    @property
    def available_permits(self) -> int:
        return max(self.permits - self.inflight, 0)

    @property
    def load(self) -> float:
        return self.inflight / max(self.permits, 1)

    @property
    def trust_tier(self) -> str:
        return str(self.metadata.get("trust_tier") or self.worker.manifest.trust_tier or "standard")

    @property
    def trust_rank(self) -> int:
        return TRUST_TIER_RANK.get(self.trust_tier, 0)

    @property
    def queue_pressure(self) -> float:
        raw = self.metadata.get("queue_pressure", self.metadata.get("queue_depth", 0.0))
        try:
            return max(float(raw), 0.0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def health(self) -> float:
        raw = self.metadata.get("health", 1.0)
        try:
            return max(min(float(raw), 1.0), 0.0)
        except (TypeError, ValueError):
            return 1.0

    def selection_key(self, minimum_trust_rank: int, backlog_depth: int = 0) -> tuple[float, float, float, str]:
        trust_surplus = max(self.trust_rank - minimum_trust_rank, 0)
        backlog_penalty = float(backlog_depth) / max(self.permits, 1)
        effective_pressure = self.queue_pressure + self.inflight + backlog_penalty
        health_penalty = 1.0 - self.health
        return (
            float(trust_surplus),
            effective_pressure + health_penalty,
            -float(self.available_permits),
            self.worker.node_id,
        )


class LocalWorkerIndex:
    """
    Local shard-owned worker index.

    Workers are bucketed by trust tier first, then scored by available permits
    and queue pressure. This keeps selection local and cheap for the shard.
    """

    def __init__(self):
        self._states: dict[str, WorkerState] = {}
        self._trust_buckets: dict[int, set[str]] = defaultdict(set)

    def register_worker(
        self,
        worker: AsyncMeshNode,
        *,
        permits: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> WorkerState:
        state = WorkerState(
            worker=worker,
            permits=max(int(permits), 1),
            metadata=dict(metadata or {}),
        )
        self.unregister_worker(worker.node_id)
        self._states[worker.node_id] = state
        self._trust_buckets[state.trust_rank].add(worker.node_id)
        return state

    def unregister_worker(self, node_id: str):
        state = self._states.pop(node_id, None)
        if state is None:
            return
        bucket = self._trust_buckets.get(state.trust_rank)
        if bucket is not None:
            bucket.discard(node_id)
            if not bucket:
                self._trust_buckets.pop(state.trust_rank, None)

    def get(self, node_id: str) -> WorkerState | None:
        return self._states.get(node_id)

    def update_worker(
        self,
        node_id: str,
        *,
        permits: int | None = None,
        inflight: int | None = None,
        metadata: dict[str, Any] | None = None,
        merge_metadata: bool = True,
    ) -> WorkerState | None:
        state = self._states.get(node_id)
        if state is None:
            return None

        old_rank = state.trust_rank
        if permits is not None:
            state.permits = max(int(permits), 1)
        if inflight is not None:
            state.inflight = max(int(inflight), 0)
        if metadata is not None:
            if merge_metadata:
                state.metadata.update(dict(metadata))
            else:
                state.metadata = dict(metadata)

        if state.trust_rank != old_rank:
            self._trust_buckets[old_rank].discard(node_id)
            if not self._trust_buckets[old_rank]:
                self._trust_buckets.pop(old_rank, None)
            self._trust_buckets[state.trust_rank].add(node_id)
        return state

    def reserve(self, node_id: str) -> bool:
        state = self._states.get(node_id)
        if state is None or not state.available:
            return False
        state.inflight += 1
        return True

    def release(self, node_id: str):
        state = self._states.get(node_id)
        if state is None:
            return
        state.inflight = max(state.inflight - 1, 0)

    def select_worker(
        self,
        task: MeshTask,
        *,
        backlog_depth: int = 0,
        predicate: Callable[[WorkerState], bool] | None = None,
    ) -> WorkerState | None:
        minimum_rank = TRUST_TIER_RANK.get(str(task.min_trust_tier or "standard"), 0)
        candidates: list[WorkerState] = []

        for rank in sorted(self._trust_buckets):
            if rank < minimum_rank:
                continue
            for node_id in self._trust_buckets[rank]:
                state = self._states.get(node_id)
                if state is None or not state.available:
                    continue
                if predicate is not None and not predicate(state):
                    continue
                candidates.append(state)

        if not candidates:
            return None

        candidates.sort(key=lambda item: item.selection_key(minimum_rank, backlog_depth))
        for state in candidates:
            allowed, _ = state.worker.can_accept_task(task)
            if allowed:
                return state
        return None

    def snapshot(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for state in sorted(self._states.values(), key=lambda item: item.worker.node_id):
            rows.append({
                "node_id": state.worker.node_id,
                "trust_tier": state.trust_tier,
                "permits": state.permits,
                "inflight": state.inflight,
                "available_permits": state.available_permits,
                "queue_pressure": state.queue_pressure,
                "health": state.health,
                "metadata": dict(state.metadata),
            })
        return rows

    def aggregate(self) -> dict[str, Any]:
        states = list(self._states.values())
        return {
            "worker_count": len(states),
            "healthy_workers": sum(1 for state in states if state.health > 0.0),
            "busy_workers": sum(1 for state in states if state.inflight > 0),
            "inflight": sum(state.inflight for state in states),
            "total_permits": sum(state.permits for state in states),
            "available_permits": sum(state.available_permits for state in states),
            "queue_pressure": sum(state.queue_pressure for state in states),
        }
