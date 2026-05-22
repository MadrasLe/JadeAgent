"""
A2A bridge helpers.

The internal Jade mesh protocol remains canonical. This module maps Jade node
and task contracts into A2A-compatible shapes for interoperability gateways.
"""

from __future__ import annotations

from typing import Any

from .governance import NodeManifest
from .mesh.protocol import MeshTask


def manifest_to_agent_card(manifest: NodeManifest) -> dict[str, Any]:
    """Project a NodeManifest into an Agent Card-like document."""
    skills = []
    for capability in manifest.capabilities:
        skills.append({
            "id": capability,
            "name": capability,
            "description": f"Jade capability '{capability}'",
            "tags": list(manifest.labels),
        })

    return {
        "id": manifest.node_id,
        "name": manifest.node_id,
        "description": manifest.description or f"Jade node {manifest.node_id}",
        "url": manifest.agent_card_url or "",
        "version": manifest.protocol_version,
        "authentication": {"schemes": list(manifest.auth_schemes)},
        "metadata": {
            "role": manifest.role,
            "tenant_id": manifest.tenant_id,
            "trust_tier": manifest.trust_tier,
            "labels": list(manifest.labels),
            "delegation_allowlist": list(manifest.delegation_allowlist),
        },
        "skills": skills,
    }


def task_to_a2a_request(task: MeshTask) -> dict[str, Any]:
    """Project a MeshTask into a simple A2A request envelope."""
    return {
        "id": task.task_id,
        "skill": task.capability,
        "input": task.prompt,
        "metadata": {
            "requester": task.requester,
            "priority": task.priority,
            "affinity": task.affinity,
            "tenant_id": task.tenant_id,
            "memory_scope": task.memory_scope,
            "parent_task_id": task.parent_task_id,
            "min_trust_tier": task.min_trust_tier,
            "max_attempts": task.max_attempts,
            "lease_seconds": task.lease_seconds,
            "task_policy": dict(task.task_policy),
            "jade_metadata": dict(task.metadata),
        },
    }


def a2a_request_to_task(
    request: dict[str, Any],
    *,
    requester: str = "a2a-client",
) -> MeshTask:
    """Translate an A2A-compatible payload back into a MeshTask."""
    metadata = request.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    return MeshTask(
        capability=str(request.get("skill", "")),
        prompt=str(request.get("input", "")),
        requester=str(metadata.get("requester", requester)),
        task_id=str(request.get("id", "")),
        priority=int(metadata.get("priority", 0)),
        affinity=metadata.get("affinity"),
        metadata=dict(metadata.get("jade_metadata", {})),
        task_policy=dict(metadata.get("task_policy", {})),
        max_attempts=int(metadata.get("max_attempts", 3)),
        lease_seconds=float(metadata.get("lease_seconds", 30.0)),
        tenant_id=str(metadata.get("tenant_id", "")),
        memory_scope=str(metadata.get("memory_scope", "")),
        parent_task_id=str(metadata.get("parent_task_id", "")) or None,
        min_trust_tier=str(metadata.get("min_trust_tier", "standard")),
    )
