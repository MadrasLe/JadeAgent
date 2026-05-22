"""
Helpers for running async mesh nodes.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Callable

from ..core.agent import Agent
from ..governance import TaskPolicy
from .agent_node import extract_mesh_answer
from .async_node import AsyncMeshNode
from .protocol import MeshTask, TaskResult, TaskState


def make_async_agent_task_handler(
    agent: Agent,
    node_id: str,
    preprocessor: Callable[[MeshTask], str] | None = None,
) -> Callable[[MeshTask], Any]:
    """
    Wrap a sync Agent for use in AsyncMeshNode execution.
    """

    async def _run(task: MeshTask) -> str:
        prompt = preprocessor(task) if preprocessor is not None else task.prompt
        task_policy = TaskPolicy.from_dict(task.task_policy)
        result = await asyncio.to_thread(
            agent.run,
            prompt,
            task_policy,
            {
                "task_id": task.task_id,
                "capability": task.capability,
                "tenant_id": task.tenant_id,
                "parent_task_id": task.parent_task_id or "",
                "memory_scope": task.memory_scope,
                "requester": task.requester,
            },
        )
        payload = {
            "worker": node_id,
            "agent": agent.name,
            "task_id": task.task_id,
            "capability": task.capability,
            "prompt": task.prompt,
            "answer": result.answer,
            "steps": result.steps,
            "tool_calls": [tc.name for tc in result.tool_calls_made],
            "tenant_id": task.tenant_id,
            "memory_scope": task.memory_scope,
            "parent_task_id": task.parent_task_id,
            "success": True,
        }
        return json.dumps(payload, ensure_ascii=True)

    return _run


class AsyncMeshDelegationClient:
    """
    Async helper used by agents or coordinators to delegate work across the mesh.
    """

    def __init__(
        self,
        router,
        bus,
        node_id: str | None = None,
        capability: str = "delegate",
        wait_route_seconds: float = 10.0,
        route_poll_interval: float = 0.2,
        default_task_ttl: int = 8,
        task_store=None,
        memory_router=None,
        audit_sink: Any = None,
        manifest=None,
    ):
        self.router = router
        self.bus = bus
        self.task_store = task_store
        self.memory_router = memory_router
        self.audit_sink = audit_sink or task_store
        self.wait_route_seconds = float(wait_route_seconds)
        self.route_poll_interval = float(route_poll_interval)
        self.default_task_ttl = int(default_task_ttl)
        self.node = AsyncMeshNode(
            node_id=node_id or f"async_mesh_delegate_{uuid.uuid4().hex[:8]}",
            capabilities={capability},
            router=router,
            bus=bus,
            task_handler=lambda task: f"delegate ack {task.task_id}",
            verbose=False,
            manifest=manifest,
            task_store=task_store,
            memory_router=memory_router,
            audit_sink=self.audit_sink,
        )

    def has_route(self, capability: str) -> bool:
        for row in self.router.snapshot():
            caps = set(row.get("capabilities", []))
            if capability in caps:
                return True
        return False

    async def wait_for_route(self, capability: str) -> bool:
        if self.has_route(capability):
            return True

        if self.wait_route_seconds <= 0:
            return False

        deadline = asyncio.get_running_loop().time() + self.wait_route_seconds
        while asyncio.get_running_loop().time() < deadline:
            if self.has_route(capability):
                return True
            await asyncio.sleep(max(self.route_poll_interval, 0.05))
        return False

    async def submit(
        self,
        capability: str,
        prompt: str,
        metadata: dict | None = None,
        task_policy: dict | None = None,
        timeout_seconds: float = 30.0,
        ttl: int | None = None,
        max_attempts: int = 3,
        lease_seconds: float = 30.0,
        tenant_id: str = "",
        memory_scope: str = "",
        parent_task_id: str | None = None,
        min_trust_tier: str = "standard",
    ) -> TaskResult:
        if not await self.wait_for_route(capability):
            result = TaskResult(
                task_id=uuid.uuid4().hex,
                capability=capability,
                node_id=self.node.node_id,
                metadata=dict(metadata or {}),
            )
            result.finalize(
                state=TaskState.FAILED,
                error=f"No route available for capability '{capability}'",
            )
            return result

        task = MeshTask(
            capability=capability,
            prompt=prompt,
            requester=self.node.node_id,
            ttl=ttl or self.default_task_ttl,
            metadata=dict(metadata or {}),
            task_policy=dict(task_policy or {}),
            max_attempts=max_attempts,
            lease_seconds=lease_seconds,
            tenant_id=tenant_id,
            memory_scope=memory_scope,
            parent_task_id=parent_task_id,
            min_trust_tier=min_trust_tier,
        )
        task_id = await self.node.submit_task(task)
        result = await self.node.wait_for_result(task_id, timeout=timeout_seconds)
        if result is not None:
            return result

        timeout_result = TaskResult(
            task_id=task_id,
            capability=capability,
            node_id=self.node.node_id,
            metadata=dict(metadata or {}),
        )
        timeout_result.finalize(
            state=TaskState.FAILED,
            error=f"Timed out waiting for capability '{capability}'",
        )
        return timeout_result

    async def submit_text(
        self,
        capability: str,
        prompt: str,
        metadata: dict | None = None,
        task_policy: dict | None = None,
        timeout_seconds: float = 30.0,
        ttl: int | None = None,
        max_attempts: int = 3,
        lease_seconds: float = 30.0,
        tenant_id: str = "",
        memory_scope: str = "",
        parent_task_id: str | None = None,
        min_trust_tier: str = "standard",
    ) -> str:
        result = await self.submit(
            capability=capability,
            prompt=prompt,
            metadata=metadata,
            task_policy=task_policy,
            timeout_seconds=timeout_seconds,
            ttl=ttl,
            max_attempts=max_attempts,
            lease_seconds=lease_seconds,
            tenant_id=tenant_id,
            memory_scope=memory_scope,
            parent_task_id=parent_task_id,
            min_trust_tier=min_trust_tier,
        )
        if result.output:
            return extract_mesh_answer(result.output)
        return result.error or f"Delegation failed for capability '{capability}'"

    async def close(self):
        await self.node.close()
