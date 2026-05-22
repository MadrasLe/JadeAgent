"""
Capability-aware router for mesh agent networks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any

from ..governance import trust_tier_allows


@dataclass
class NodeState:
    node_id: str
    capabilities: set[str]
    max_inflight: int = 4
    inflight: int = 0
    queue_depth: int = 0
    last_seen: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def load_factor(self) -> float:
        denom = max(self.max_inflight, 1)
        return (self.inflight + self.queue_depth) / denom

    @property
    def available_slots(self) -> int:
        slots = self.max_inflight - self.inflight - self.queue_depth
        return slots if slots > 0 else 0


def _tenant_compatible(node_meta: dict[str, Any], tenant_id: str | None) -> bool:
    if not tenant_id:
        return True
    node_tenant = str(node_meta.get("tenant_id", ""))
    return node_tenant in ("", tenant_id)


def _delegation_allowed(node_meta: dict[str, Any], requester: str | None) -> bool:
    allowlist = node_meta.get("delegation_allowlist", [])
    if not allowlist:
        return True
    if not requester:
        return False
    return any(fnmatch(str(requester), str(pattern)) for pattern in allowlist)


class MeshRouter:
    """
    In-memory capability router for task placement.
    """

    def __init__(self, stale_after: float = 30.0):
        self.stale_after = stale_after
        self._nodes: dict[str, NodeState] = {}

    def register_node(
        self,
        node_id: str,
        capabilities: set[str] | list[str] | tuple[str, ...],
        max_inflight: int = 4,
        metadata: dict[str, Any] | None = None,
    ):
        self._nodes[node_id] = NodeState(
            node_id=node_id,
            capabilities=set(capabilities),
            max_inflight=max_inflight,
            metadata=dict(metadata or {}),
        )

    def unregister_node(self, node_id: str):
        self._nodes.pop(node_id, None)

    def update_heartbeat(self, node_id: str, queue_depth: int | None = None):
        state = self._nodes.get(node_id)
        if state is None:
            return
        state.last_seen = time.time()
        if queue_depth is not None:
            state.queue_depth = max(queue_depth, 0)

    def mark_assigned(self, node_id: str):
        state = self._nodes.get(node_id)
        if state is not None:
            state.inflight += 1

    def mark_done(self, node_id: str):
        state = self._nodes.get(node_id)
        if state is not None and state.inflight > 0:
            state.inflight -= 1

    def route(
        self,
        capability: str,
        exclude: set[str] | None = None,
        affinity: str | None = None,
        tenant_id: str | None = None,
        min_trust_tier: str | None = None,
        requester: str | None = None,
    ) -> str | None:
        exclude = exclude or set()
        now = time.time()

        candidates: list[NodeState] = []
        for state in self._nodes.values():
            if state.node_id in exclude:
                continue
            if capability not in state.capabilities:
                continue
            if now - state.last_seen > self.stale_after:
                continue
            if not _tenant_compatible(state.metadata, tenant_id):
                continue
            if not trust_tier_allows(str(state.metadata.get("trust_tier", "standard")), min_trust_tier):
                continue
            if not _delegation_allowed(state.metadata, requester):
                continue
            candidates.append(state)

        if not candidates:
            return None

        if affinity:
            ordered = sorted(candidates, key=lambda s: s.node_id)
            idx = sum(ord(ch) for ch in affinity) % len(ordered)
            return ordered[idx].node_id

        def score(s: NodeState) -> tuple[float, float, str]:
            freshness = -(now - s.last_seen)
            return (-s.load_factor, freshness, s.node_id)

        return max(candidates, key=score).node_id

    def snapshot(self) -> list[dict[str, Any]]:
        now = time.time()
        rows = []
        for state in sorted(self._nodes.values(), key=lambda s: s.node_id):
            rows.append({
                "node_id": state.node_id,
                "capabilities": sorted(state.capabilities),
                "inflight": state.inflight,
                "queue_depth": state.queue_depth,
                "max_inflight": state.max_inflight,
                "load_factor": round(state.load_factor, 3),
                "age_seconds": round(now - state.last_seen, 3),
                "metadata": dict(state.metadata),
            })
        return rows

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)
