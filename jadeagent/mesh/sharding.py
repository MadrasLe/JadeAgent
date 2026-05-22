"""
Shard directory and deterministic placement primitives.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

from .protocol import MeshTask


def shard_key(tenant_id: str, capability: str) -> str:
    tenant = tenant_id or "default"
    return f"{tenant}::{capability}"


def _rendezvous_score(key: str, supervisor_id: str, weight: float = 1.0) -> float:
    digest = hashlib.sha256(f"{key}::{supervisor_id}".encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], "big")
    unit = (raw + 1) / float(2**64)
    effective_weight = max(float(weight), 1e-9)
    return effective_weight / -math.log(unit)


@dataclass(frozen=True)
class SupervisorSpec:
    supervisor_id: str
    capabilities: tuple[str, ...] = ()
    tenants: tuple[str, ...] = ()
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches(self, tenant_id: str, capability: str) -> bool:
        if self.capabilities and capability not in self.capabilities:
            return False
        if self.tenants and tenant_id not in self.tenants:
            return False
        return True


@dataclass(frozen=True)
class ShardAssignment:
    shard_id: str
    shard_key: str
    supervisor_id: str
    tenant_id: str
    capability: str
    score: float


class ShardDirectory:
    """
    Deterministic shard directory using rendezvous hashing.
    """

    def __init__(self):
        self._supervisors: dict[str, SupervisorSpec] = {}

    def register_supervisor(
        self,
        supervisor_id: str,
        *,
        capabilities: set[str] | list[str] | tuple[str, ...] | None = None,
        tenants: set[str] | list[str] | tuple[str, ...] | None = None,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ):
        self._supervisors[supervisor_id] = SupervisorSpec(
            supervisor_id=supervisor_id,
            capabilities=tuple(str(item) for item in (capabilities or ())),
            tenants=tuple(str(item) for item in (tenants or ())),
            weight=float(weight),
            metadata=dict(metadata or {}),
        )

    def unregister_supervisor(self, supervisor_id: str):
        self._supervisors.pop(supervisor_id, None)

    def route(self, tenant_id: str, capability: str) -> ShardAssignment | None:
        key = shard_key(tenant_id, capability)
        best: tuple[float, SupervisorSpec] | None = None

        for spec in self._supervisors.values():
            if not spec.matches(tenant_id, capability):
                continue
            score = _rendezvous_score(key, spec.supervisor_id, spec.weight)
            if best is None or score > best[0]:
                best = (score, spec)

        if best is None:
            return None

        score, spec = best
        return ShardAssignment(
            shard_id=f"shard::{key}::{spec.supervisor_id}",
            shard_key=key,
            supervisor_id=spec.supervisor_id,
            tenant_id=tenant_id,
            capability=capability,
            score=score,
        )

    def route_task(self, task: MeshTask) -> ShardAssignment | None:
        return self.route(task.tenant_id, task.capability)

    def snapshot(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for spec in sorted(self._supervisors.values(), key=lambda item: item.supervisor_id):
            rows.append({
                "supervisor_id": spec.supervisor_id,
                "capabilities": list(spec.capabilities),
                "tenants": list(spec.tenants),
                "weight": spec.weight,
                "metadata": dict(spec.metadata),
            })
        return rows
