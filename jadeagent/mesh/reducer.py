"""
Reduction layer for shard summaries, tenant rollups, and batch metrics.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, TypeVar


T = TypeVar("T")


def hillis_steele_scan(values: Iterable[T], op: Callable[[T, T], T] = operator.add) -> list[T]:
    """
    Inclusive prefix scan using the Hillis-Steele update pattern.
    """
    result = list(values)
    step = 1
    while step < len(result):
        prev = list(result)
        for index in range(step, len(result)):
            result[index] = op(prev[index - step], prev[index])
        step <<= 1
    return result


def hillis_steele_reduce(
    values: Iterable[T],
    op: Callable[[T, T], T] = operator.add,
    identity: T | None = None,
) -> T | None:
    items = list(values)
    if not items:
        return identity
    return hillis_steele_scan(items, op=op)[-1]


@dataclass(frozen=True)
class ShardSummary:
    shard_id: str
    supervisor_id: str
    tenant_id: str
    capability: str
    ready_depth: int = 0
    retry_depth: int = 0
    inflight: int = 0
    dead_letter: int = 0
    completed: int = 0
    worker_count: int = 0
    healthy_workers: int = 0
    busy_workers: int = 0
    total_permits: int = 0
    available_permits: int = 0
    queue_pressure: float = 0.0
    event_counts: dict[str, int] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def queued(self) -> int:
        return self.ready_depth + self.retry_depth

    @property
    def failed(self) -> int:
        return int(self.event_counts.get("task_failed", 0))

    @property
    def policy_denied(self) -> int:
        return int(self.event_counts.get("policy_denied", 0))

    @property
    def lease_expired(self) -> int:
        return int(self.event_counts.get("lease_expired", 0))

    @classmethod
    def from_supervisor_snapshot(cls, snapshot: dict[str, Any]) -> ShardSummary:
        worker_metrics = dict(snapshot.get("worker_metrics", {}))
        stats = {str(key): int(value) for key, value in dict(snapshot.get("stats", {})).items()}
        return cls(
            shard_id=str(snapshot.get("shard_id") or f"{snapshot.get('tenant_id', 'default')}::{snapshot.get('capability', '')}"),
            supervisor_id=str(snapshot.get("supervisor_id", "")),
            tenant_id=str(snapshot.get("tenant_id", "")),
            capability=str(snapshot.get("capability", "")),
            ready_depth=int(snapshot.get("ready_depth", 0)),
            retry_depth=int(snapshot.get("retry_depth", 0)),
            inflight=int(worker_metrics.get("inflight", 0)),
            dead_letter=int(snapshot.get("dead_letter", 0)),
            completed=int(snapshot.get("completed", 0)),
            worker_count=int(worker_metrics.get("worker_count", 0)),
            healthy_workers=int(worker_metrics.get("healthy_workers", 0)),
            busy_workers=int(worker_metrics.get("busy_workers", 0)),
            total_permits=int(worker_metrics.get("total_permits", 0)),
            available_permits=int(worker_metrics.get("available_permits", 0)),
            queue_pressure=float(snapshot.get("ready_depth", 0))
            + float(snapshot.get("retry_depth", 0))
            + float(worker_metrics.get("queue_pressure", 0.0)),
            event_counts=stats,
            created_at=float(snapshot.get("last_updated_at", time.time())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "supervisor_id": self.supervisor_id,
            "tenant_id": self.tenant_id,
            "capability": self.capability,
            "ready_depth": self.ready_depth,
            "retry_depth": self.retry_depth,
            "inflight": self.inflight,
            "dead_letter": self.dead_letter,
            "completed": self.completed,
            "worker_count": self.worker_count,
            "healthy_workers": self.healthy_workers,
            "busy_workers": self.busy_workers,
            "total_permits": self.total_permits,
            "available_permits": self.available_permits,
            "queue_pressure": self.queue_pressure,
            "event_counts": dict(self.event_counts),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class TenantBudgetSummary:
    tenant_id: str
    queued: int = 0
    inflight: int = 0
    completed: int = 0
    failed: int = 0
    dead_letter: int = 0
    policy_denied: int = 0
    lease_expired: int = 0
    total_permits: int = 0
    available_permits: int = 0
    queue_pressure: float = 0.0
    headroom: int = 0
    budget_pressure: float = 0.0

    @classmethod
    def from_shard(cls, summary: ShardSummary) -> TenantBudgetSummary:
        return cls(
            tenant_id=summary.tenant_id,
            queued=summary.queued,
            inflight=summary.inflight,
            completed=summary.completed,
            failed=summary.failed,
            dead_letter=summary.dead_letter,
            policy_denied=summary.policy_denied,
            lease_expired=summary.lease_expired,
            total_permits=summary.total_permits,
            available_permits=summary.available_permits,
            queue_pressure=summary.queue_pressure,
            headroom=summary.available_permits,
            budget_pressure=(summary.queued + summary.inflight) / max(summary.total_permits, 1),
        )

    def merge(self, other: TenantBudgetSummary) -> TenantBudgetSummary:
        total_permits = self.total_permits + other.total_permits
        available_permits = self.available_permits + other.available_permits
        queued = self.queued + other.queued
        inflight = self.inflight + other.inflight
        return TenantBudgetSummary(
            tenant_id=self.tenant_id,
            queued=queued,
            inflight=inflight,
            completed=self.completed + other.completed,
            failed=self.failed + other.failed,
            dead_letter=self.dead_letter + other.dead_letter,
            policy_denied=self.policy_denied + other.policy_denied,
            lease_expired=self.lease_expired + other.lease_expired,
            total_permits=total_permits,
            available_permits=available_permits,
            queue_pressure=self.queue_pressure + other.queue_pressure,
            headroom=available_permits,
            budget_pressure=(queued + inflight) / max(total_permits, 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "queued": self.queued,
            "inflight": self.inflight,
            "completed": self.completed,
            "failed": self.failed,
            "dead_letter": self.dead_letter,
            "policy_denied": self.policy_denied,
            "lease_expired": self.lease_expired,
            "total_permits": self.total_permits,
            "available_permits": self.available_permits,
            "queue_pressure": self.queue_pressure,
            "headroom": self.headroom,
            "budget_pressure": self.budget_pressure,
        }


@dataclass(frozen=True)
class ReductionSummary:
    reducer_id: str
    region: str = ""
    shard_count: int = 0
    child_count: int = 0
    total_queued: int = 0
    total_inflight: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_dead_letter: int = 0
    total_workers: int = 0
    healthy_workers: int = 0
    busy_workers: int = 0
    total_permits: int = 0
    available_permits: int = 0
    queue_pressure_total: float = 0.0
    queue_pressure_prefix: tuple[float, ...] = ()
    event_totals: dict[str, int] = field(default_factory=dict)
    tenant_budgets: dict[str, TenantBudgetSummary] = field(default_factory=dict)
    sources: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reducer_id": self.reducer_id,
            "region": self.region,
            "shard_count": self.shard_count,
            "child_count": self.child_count,
            "total_queued": self.total_queued,
            "total_inflight": self.total_inflight,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "total_dead_letter": self.total_dead_letter,
            "total_workers": self.total_workers,
            "healthy_workers": self.healthy_workers,
            "busy_workers": self.busy_workers,
            "total_permits": self.total_permits,
            "available_permits": self.available_permits,
            "queue_pressure_total": self.queue_pressure_total,
            "queue_pressure_prefix": list(self.queue_pressure_prefix),
            "event_totals": dict(self.event_totals),
            "tenant_budgets": {
                tenant_id: summary.to_dict()
                for tenant_id, summary in sorted(self.tenant_budgets.items())
            },
            "sources": list(self.sources),
            "created_at": self.created_at,
        }


class ReducerNode:
    """
    Hierarchical reducer for shard summaries and tenant rollups.
    """

    def __init__(self, reducer_id: str, *, region: str = ""):
        self.reducer_id = reducer_id
        self.region = region
        self._shards: dict[str, ShardSummary] = {}
        self._children: dict[str, ReductionSummary] = {}

    def ingest_shard(self, summary: ShardSummary):
        self._shards[summary.supervisor_id] = summary

    def ingest_child(self, summary: ReductionSummary):
        self._children[summary.reducer_id] = summary

    def remove_shard(self, supervisor_id: str):
        self._shards.pop(supervisor_id, None)

    def remove_child(self, reducer_id: str):
        self._children.pop(reducer_id, None)

    def clear(self):
        self._shards.clear()
        self._children.clear()

    def reduce(self) -> ReductionSummary:
        shard_summaries = sorted(self._shards.values(), key=lambda item: (item.tenant_id, item.capability, item.supervisor_id))
        child_summaries = sorted(self._children.values(), key=lambda item: item.reducer_id)

        queue_pressure_inputs = [float(summary.queue_pressure) for summary in shard_summaries]
        queue_pressure_inputs.extend(float(summary.queue_pressure_total) for summary in child_summaries)
        queue_pressure_prefix = tuple(hillis_steele_scan(queue_pressure_inputs))
        queue_pressure_total = float(queue_pressure_prefix[-1]) if queue_pressure_prefix else 0.0

        total_queued = int(hillis_steele_reduce(
            [summary.queued for summary in shard_summaries] + [summary.total_queued for summary in child_summaries],
            identity=0,
        ) or 0)
        total_inflight = int(hillis_steele_reduce(
            [summary.inflight for summary in shard_summaries] + [summary.total_inflight for summary in child_summaries],
            identity=0,
        ) or 0)
        total_completed = int(hillis_steele_reduce(
            [summary.completed for summary in shard_summaries] + [summary.total_completed for summary in child_summaries],
            identity=0,
        ) or 0)
        total_failed = int(hillis_steele_reduce(
            [summary.failed for summary in shard_summaries] + [summary.total_failed for summary in child_summaries],
            identity=0,
        ) or 0)
        total_dead_letter = int(hillis_steele_reduce(
            [summary.dead_letter for summary in shard_summaries] + [summary.total_dead_letter for summary in child_summaries],
            identity=0,
        ) or 0)
        total_workers = int(hillis_steele_reduce(
            [summary.worker_count for summary in shard_summaries] + [summary.total_workers for summary in child_summaries],
            identity=0,
        ) or 0)
        healthy_workers = int(hillis_steele_reduce(
            [summary.healthy_workers for summary in shard_summaries] + [summary.healthy_workers for summary in child_summaries],
            identity=0,
        ) or 0)
        busy_workers = int(hillis_steele_reduce(
            [summary.busy_workers for summary in shard_summaries] + [summary.busy_workers for summary in child_summaries],
            identity=0,
        ) or 0)
        total_permits = int(hillis_steele_reduce(
            [summary.total_permits for summary in shard_summaries] + [summary.total_permits for summary in child_summaries],
            identity=0,
        ) or 0)
        available_permits = int(hillis_steele_reduce(
            [summary.available_permits for summary in shard_summaries] + [summary.available_permits for summary in child_summaries],
            identity=0,
        ) or 0)

        event_totals: dict[str, int] = {}
        for summary in shard_summaries:
            for event_type, count in summary.event_counts.items():
                event_totals[event_type] = event_totals.get(event_type, 0) + int(count)
        for summary in child_summaries:
            for event_type, count in summary.event_totals.items():
                event_totals[event_type] = event_totals.get(event_type, 0) + int(count)

        tenant_budgets: dict[str, TenantBudgetSummary] = {}
        for summary in shard_summaries:
            tenant_id = summary.tenant_id or "default"
            shard_budget = TenantBudgetSummary.from_shard(summary)
            existing = tenant_budgets.get(tenant_id)
            tenant_budgets[tenant_id] = shard_budget if existing is None else existing.merge(shard_budget)
        for summary in child_summaries:
            for tenant_id, tenant_budget in summary.tenant_budgets.items():
                existing = tenant_budgets.get(tenant_id)
                tenant_budgets[tenant_id] = tenant_budget if existing is None else existing.merge(tenant_budget)

        sources = tuple(
            list(summary.supervisor_id for summary in shard_summaries)
            + list(summary.reducer_id for summary in child_summaries)
        )

        return ReductionSummary(
            reducer_id=self.reducer_id,
            region=self.region,
            shard_count=len(shard_summaries),
            child_count=len(child_summaries),
            total_queued=total_queued,
            total_inflight=total_inflight,
            total_completed=total_completed,
            total_failed=total_failed,
            total_dead_letter=total_dead_letter,
            total_workers=total_workers,
            healthy_workers=healthy_workers,
            busy_workers=busy_workers,
            total_permits=total_permits,
            available_permits=available_permits,
            queue_pressure_total=queue_pressure_total,
            queue_pressure_prefix=queue_pressure_prefix,
            event_totals=dict(sorted(event_totals.items())),
            tenant_budgets={tenant_id: tenant_budgets[tenant_id] for tenant_id in sorted(tenant_budgets)},
            sources=sources,
        )
