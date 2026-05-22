"""
Distributed mesh router with Redis-backed node discovery.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .router import NodeState, _delegation_allowed, _tenant_compatible
from ..governance import trust_tier_allows


class DistributedMeshRouter:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        registry_prefix: str = "jade:mesh:registry",
        stale_after: float = 30.0,
        heartbeat_ttl: int | None = None,
        refresh_interval: float = 0.25,
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
                "DistributedMeshRouter requires redis package. Install with: pip install redis"
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

        self.registry_prefix = registry_prefix.rstrip(":")
        self.stale_after = stale_after
        self.heartbeat_ttl = heartbeat_ttl or int(max(stale_after * 3, 10))
        self.refresh_interval = refresh_interval

        self._local_nodes: dict[str, NodeState] = {}
        self._cache: dict[str, NodeState] = {}
        self._last_refresh: float = 0.0

    @property
    def _nodes_key(self) -> str:
        return f"{self.registry_prefix}:nodes"

    def _node_key(self, node_id: str) -> str:
        return f"{self.registry_prefix}:node:{node_id}"

    def register_node(
        self,
        node_id: str,
        capabilities: set[str] | list[str] | tuple[str, ...],
        max_inflight: int = 4,
        metadata: dict[str, Any] | None = None,
    ):
        state = NodeState(
            node_id=node_id,
            capabilities=set(capabilities),
            max_inflight=max_inflight,
            metadata=dict(metadata or {}),
        )
        self._local_nodes[node_id] = state
        self._publish_state(state)
        self._cache[node_id] = state

    def unregister_node(self, node_id: str):
        self._local_nodes.pop(node_id, None)
        self._cache.pop(node_id, None)

        pipe = self._client.pipeline(transaction=False)
        pipe.srem(self._nodes_key, node_id)
        pipe.delete(self._node_key(node_id))
        pipe.execute()

    def update_heartbeat(self, node_id: str, queue_depth: int | None = None):
        state = self._local_nodes.get(node_id)
        if state is None:
            return
        state.last_seen = time.time()
        if queue_depth is not None:
            state.queue_depth = max(int(queue_depth), 0)
        self._publish_state(state)
        self._cache[node_id] = state

    def mark_assigned(self, node_id: str):
        state = self._local_nodes.get(node_id)
        if state is None:
            return
        state.inflight += 1
        state.last_seen = time.time()
        self._publish_state(state)
        self._cache[node_id] = state

    def mark_done(self, node_id: str):
        state = self._local_nodes.get(node_id)
        if state is None:
            return
        if state.inflight > 0:
            state.inflight -= 1
        state.last_seen = time.time()
        self._publish_state(state)
        self._cache[node_id] = state

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
        candidates = [
            state
            for state in self._discover_nodes().values()
            if state.node_id not in exclude
            and capability in state.capabilities
            and _tenant_compatible(state.metadata, tenant_id)
            and trust_tier_allows(str(state.metadata.get("trust_tier", "standard")), min_trust_tier)
            and _delegation_allowed(state.metadata, requester)
        ]
        if not candidates:
            return None

        if affinity:
            ordered = sorted(candidates, key=lambda s: s.node_id)
            idx = sum(ord(ch) for ch in affinity) % len(ordered)
            return ordered[idx].node_id

        now = time.time()

        def score(s: NodeState) -> tuple[float, float, str]:
            freshness = -(now - s.last_seen)
            return (-s.load_factor, freshness, s.node_id)

        return max(candidates, key=score).node_id

    def snapshot(self) -> list[dict[str, Any]]:
        nodes = self._discover_nodes()
        now = time.time()
        rows = []
        for state in sorted(nodes.values(), key=lambda s: s.node_id):
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

    def _discover_nodes(self) -> dict[str, NodeState]:
        now = time.time()
        if now - self._last_refresh < self.refresh_interval:
            return self._cache

        node_ids = [str(x) for x in self._client.smembers(self._nodes_key)]
        if not node_ids:
            self._cache = {}
            self._last_refresh = now
            return self._cache

        pipe = self._client.pipeline(transaction=False)
        for node_id in node_ids:
            pipe.hgetall(self._node_key(node_id))
        rows = pipe.execute()

        discovered: dict[str, NodeState] = {}
        for node_id, row in zip(node_ids, rows):
            if not row:
                continue
            state = self._parse_state(node_id, row)
            if state is None:
                continue
            if now - state.last_seen > self.stale_after:
                continue
            discovered[node_id] = state

        for node_id, state in self._local_nodes.items():
            discovered[node_id] = state

        self._cache = discovered
        self._last_refresh = now
        return self._cache

    def _publish_state(self, state: NodeState):
        payload = {
            "node_id": state.node_id,
            "capabilities": json.dumps(sorted(state.capabilities), separators=(",", ":")),
            "max_inflight": str(int(state.max_inflight)),
            "inflight": str(int(state.inflight)),
            "queue_depth": str(int(state.queue_depth)),
            "last_seen": str(float(state.last_seen)),
            "metadata": json.dumps(state.metadata, separators=(",", ":")),
        }

        key = self._node_key(state.node_id)
        pipe = self._client.pipeline(transaction=False)
        pipe.sadd(self._nodes_key, state.node_id)
        pipe.hset(key, mapping=payload)
        pipe.expire(key, self.heartbeat_ttl)
        pipe.execute()

    @staticmethod
    def _parse_state(node_id: str, row: dict[str, str]) -> NodeState | None:
        try:
            caps = set(json.loads(row.get("capabilities", "[]")))
            max_inflight = int(row.get("max_inflight", "4"))
            inflight = int(row.get("inflight", "0"))
            queue_depth = int(row.get("queue_depth", "0"))
            last_seen = float(row.get("last_seen", "0"))
            metadata = json.loads(row.get("metadata", "{}"))
            if not isinstance(metadata, dict):
                metadata = {}
        except Exception:
            return None

        return NodeState(
            node_id=node_id,
            capabilities=caps,
            max_inflight=max_inflight,
            inflight=inflight,
            queue_depth=queue_depth,
            last_seen=last_seen,
            metadata=metadata,
        )

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._discover_nodes()

    def __len__(self) -> int:
        return len(self._discover_nodes())
