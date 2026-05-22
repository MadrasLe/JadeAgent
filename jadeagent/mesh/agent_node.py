"""
Helpers for running LLM agents as mesh nodes.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable

from ..core.agent import Agent
from ..core.tools import Tool, tool
from ..governance import TaskPolicy
from .node import MeshNode
from .protocol import MeshTask, TaskResult, TaskState


def make_agent_task_handler(
    agent: Agent,
    node_id: str,
    preprocessor: Callable[[MeshTask], str] | None = None,
) -> Callable[[MeshTask], str]:
    """
    Wrap a local Agent so it can serve as a MeshNode task handler.

    The handler returns structured JSON so coordinator nodes can inspect which
    agent/capability handled a task.
    """

    def _run(task: MeshTask) -> str:
        prompt = preprocessor(task) if preprocessor is not None else task.prompt
        task_policy = TaskPolicy.from_dict(task.task_policy)
        result = agent.run(
            prompt,
            task_policy=task_policy,
            task_context={
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


def extract_mesh_answer(raw_text: str) -> str:
    """Extract the main answer text from a structured mesh JSON payload."""
    try:
        payload = json.loads(raw_text)
    except Exception:
        return raw_text

    if not isinstance(payload, dict):
        return raw_text

    for key in ("answer", "output", "stdout", "result"):
        value = payload.get(key)
        if value:
            return str(value)

    sandbox = payload.get("sandbox")
    if isinstance(sandbox, dict):
        stdout = sandbox.get("stdout")
        if stdout:
            return str(stdout)

    return raw_text


class MeshDelegationClient:
    """
    Helper node used by agents to synchronously delegate work across the mesh.

    It owns its own ephemeral MeshNode so an agent can submit downstream tasks
    without re-entering the currently executing node.
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
        self.node = MeshNode(
            node_id=node_id or f"mesh_delegate_{uuid.uuid4().hex[:8]}",
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

    def wait_for_route(self, capability: str) -> bool:
        if self.has_route(capability):
            return True

        if self.wait_route_seconds <= 0:
            return False

        deadline = time.time() + self.wait_route_seconds
        while time.time() < deadline:
            self.node.step()
            if self.has_route(capability):
                return True
            time.sleep(max(self.route_poll_interval, 0.05))
        return False

    def submit(
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
        if not self.wait_for_route(capability):
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
        task_id = self.node.submit_task(task)

        deadline = time.time() + max(float(timeout_seconds), 0.1)
        while time.time() < deadline:
            run_until_idle = getattr(self.bus, "run_until_idle", None)
            if callable(run_until_idle):
                try:
                    run_until_idle()
                except Exception:
                    pass
            self.node.step()
            result = self.node.get_result(task_id)
            if result is not None:
                return result
            time.sleep(0.05)

        result = TaskResult(
            task_id=task_id,
            capability=capability,
            node_id=self.node.node_id,
            metadata=dict(metadata or {}),
        )
        result.finalize(
            state=TaskState.FAILED,
            error=f"Timed out waiting for capability '{capability}'",
        )
        return result

    def submit_text(
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
        result = self.submit(
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

    def close(self):
        unregister_router = getattr(self.router, "unregister_node", None)
        if callable(unregister_router):
            try:
                unregister_router(self.node.node_id)
            except Exception:
                pass

        unregister_bus = getattr(self.bus, "unregister", None)
        if callable(unregister_bus):
            try:
                unregister_bus(self.node.node_id)
            except Exception:
                pass


def make_mesh_delegate_tool(
    submit_task: Callable[[str, str, dict | None, dict | None], str],
    capability: str,
    name: str | None = None,
    description: str | None = None,
) -> Tool:
    """
    Build a tool that delegates work to another capability in the mesh.

    `submit_task` should synchronously return the downstream answer/result text.
    """

    tool_name = name or f"delegate_{capability}"
    tool_description = description or f"Delegate a task to mesh capability '{capability}'."

    @tool(
        name=tool_name,
        description=tool_description,
        effects=["delegate"],
        resource_refs=[f"delegate.capability:{capability}"],
        metadata={"delegate_capability": capability},
    )
    def _delegate(prompt: str) -> str:
        try:
            return submit_task(capability, prompt, None, None)
        except TypeError:
            return submit_task(capability, prompt, None)

    return _delegate
