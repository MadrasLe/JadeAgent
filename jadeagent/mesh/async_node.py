"""
Async mesh node runtime.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, replace
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any, Callable

from .async_task_store import AsyncTaskStore, adapt_task_store
from .async_transport import AsyncMeshTransport, AsyncMeshTransportAdapter
from .protocol import (
    EnvelopeType,
    MeshTask,
    TaskResult,
    TaskState,
    make_result_envelope,
    make_task_envelope,
)
from .router import MeshRouter
from ..governance import NodeManifest, TaskPolicy, trust_tier_allows
from ..state.events import JadeStateEvent
from ..state.manifest import JadeStateManifest, canonical_json_hash
from ..state.snapshot import AgentRuntimeSnapshot, MeshRuntimeSnapshot
from ..state.store import StateStore

if TYPE_CHECKING:
    from ..core.agent import Agent
    from ..memory.router import MemoryRouter
    from .task_store import TaskRecord, TaskStore
    from .transport import MeshTransport

logger = logging.getLogger("jadeagent.mesh.async")

AsyncTaskHandler = Callable[[MeshTask], Any]


@dataclass
class AsyncNodeMetrics:
    received: int = 0
    forwarded: int = 0
    completed: int = 0
    failed: int = 0
    claimed: int = 0


class AsyncMeshNode:
    """
    Async-native mesh node.

    This complements the existing sync MeshNode and provides an event-driven
    execution path that can await transport and task-store readiness directly.
    """

    def __init__(
        self,
        node_id: str,
        capabilities: set[str] | list[str] | tuple[str, ...],
        router: MeshRouter,
        bus: AsyncMeshTransport | MeshTransport,
        agent: Agent | None = None,
        task_handler: AsyncTaskHandler | None = None,
        max_inflight: int = 2,
        verbose: bool = False,
        manifest: NodeManifest | None = None,
        task_store: AsyncTaskStore | TaskStore | None = None,
        memory_router: MemoryRouter | None = None,
        audit_sink: Any = None,
        state_store: StateStore | None = None,
    ):
        self.node_id = node_id
        self.capabilities = set(capabilities)
        self.router = router
        if hasattr(bus, "recv") and inspect.iscoroutinefunction(getattr(bus, "recv")):
            self.bus = bus
        else:
            self.bus = AsyncMeshTransportAdapter(bus)
        self.agent = agent
        self.task_handler = task_handler
        self.max_inflight = max_inflight
        self.verbose = verbose
        self.task_store = adapt_task_store(task_store) if task_store is not None else None
        self.memory_router = memory_router
        self.audit_sink = audit_sink or self.task_store
        self.state_store = state_store
        self._state_runs: set[str] = set()
        self.manifest = manifest or NodeManifest(
            node_id=node_id,
            role="worker",
            capabilities=tuple(sorted(self.capabilities)),
        )
        if not self.manifest.capabilities:
            self.manifest = replace(
                self.manifest,
                capabilities=tuple(sorted(self.capabilities)),
            )

        self._results: dict[str, TaskResult] = {}
        self._result_events: dict[str, asyncio.Event] = {}
        self._metrics = AsyncNodeMetrics()
        self._started = False

        self.router.register_node(
            node_id=self.node_id,
            capabilities=self.capabilities,
            max_inflight=self.max_inflight,
            metadata=self.manifest.routing_metadata(),
        )

    async def _ensure_started(self):
        if self._started:
            return
        await self.bus.register(self)
        self.heartbeat()
        self._started = True

    async def start(self):
        """Register the node on its async transport and publish initial liveness."""
        await self._ensure_started()

    async def close(self):
        if self._started:
            await self.bus.unregister(self.node_id)
            self._started = False
        unregister_router = getattr(self.router, "unregister_node", None)
        if callable(unregister_router):
            unregister_router(self.node_id)

    @property
    def metrics(self) -> dict[str, int]:
        return {
            "received": self._metrics.received,
            "forwarded": self._metrics.forwarded,
            "completed": self._metrics.completed,
            "failed": self._metrics.failed,
            "claimed": self._metrics.claimed,
        }

    def heartbeat(self):
        self.router.update_heartbeat(self.node_id, queue_depth=0)

    def _result_event_for(self, task_id: str) -> asyncio.Event:
        event = self._result_events.get(task_id)
        if event is None:
            event = asyncio.Event()
            self._result_events[task_id] = event
        return event

    async def submit_task(self, task: MeshTask) -> str:
        await self._ensure_started()
        if task.requester == "client":
            task.requester = self.node_id
        if not task.tenant_id:
            task.tenant_id = self.manifest.tenant_id
        if task.affinity is None:
            task.affinity = self._infer_affinity(task)

        self._result_event_for(task.task_id)

        if self.task_store is not None:
            await self.task_store.submit(task)
            await self._record_audit("task_submitted", task=task, message="task submitted")
            return task.task_id

        envelope = make_task_envelope(task, source=self.node_id)
        target = self.router.route(
            task.capability,
            affinity=task.affinity,
            tenant_id=task.tenant_id,
            min_trust_tier=task.min_trust_tier,
            requester=task.requester,
        )
        if target is None:
            result = TaskResult(
                task_id=task.task_id,
                capability=task.capability,
                node_id=self.node_id,
                metadata=dict(task.metadata),
            )
            result.finalize(
                state=TaskState.FAILED,
                error=f"No route available for capability '{task.capability}'",
            )
            self._results[task.task_id] = result
            self._result_event_for(task.task_id).set()
            self._metrics.failed += 1
            await self._record_audit("task_failed", task=task, message=result.error or "")
            return task.task_id

        envelope.destination = target
        await self.bus.send(envelope)
        await self._record_audit("task_submitted", task=task, message="task submitted")
        return task.task_id

    async def get_result(self, task_id: str) -> TaskResult | None:
        if self.task_store is not None:
            record = await self.task_store.get(task_id)
            if record is not None and record.state in {
                TaskState.COMPLETED,
                TaskState.FAILED,
                TaskState.CANCELLED,
            }:
                return record.to_result()
        return self._results.get(task_id)

    async def wait_for_result(self, task_id: str, timeout: float | None = None) -> TaskResult | None:
        if self.task_store is not None:
            record = await self.task_store.wait_for_terminal(task_id, timeout=timeout)
            return None if record is None else record.to_result()

        result = self._results.get(task_id)
        if result is not None:
            return result

        event = self._result_event_for(task_id)
        try:
            if timeout is None:
                await event.wait()
            else:
                await asyncio.wait_for(event.wait(), timeout=max(float(timeout), 0.0))
        except asyncio.TimeoutError:
            return None
        return self._results.get(task_id)

    def can_accept_task(self, task: MeshTask) -> tuple[bool, str]:
        """Public capability/policy check for shard-local scheduling."""
        return self._can_execute_task(task)

    async def execute_assigned_task(self, task: MeshTask) -> TaskResult:
        """
        Execute a task assigned by a shard supervisor without using transport.
        """
        allowed, reason = self._can_execute_task(task)
        if not allowed:
            result = TaskResult(task_id=task.task_id, capability=task.capability, node_id=self.node_id)
            result.finalize(TaskState.FAILED, error=reason)
            self._metrics.failed += 1
            await self._record_audit("task_failed", task=task, message=reason)
            await self._checkpoint_mesh_task(task, "FAILED", error=reason)
            return result

        started_at = time.time()
        result = TaskResult(
            task_id=task.task_id,
            capability=task.capability,
            node_id=self.node_id,
            started_at=started_at,
            metadata={
                "tenant_id": task.tenant_id,
                "memory_scope": task.memory_scope,
            },
        )

        self.router.mark_assigned(self.node_id)
        await self._record_audit("task_claimed", task=task, message="task claimed")
        await self._checkpoint_mesh_task(task, "RUNNING")
        try:
            output = await self._execute_task(task)
            result.finalize(TaskState.COMPLETED, output=output)
            self._metrics.completed += 1
            await self._record_audit("task_completed", task=task, message="task completed")
            await self._checkpoint_mesh_task(task, "COMPLETED", result=result)
        except Exception as exc:
            result.finalize(TaskState.FAILED, error=str(exc))
            self._metrics.failed += 1
            await self._record_audit("task_failed", task=task, message=str(exc))
            await self._checkpoint_mesh_task(task, "FAILED", error=str(exc))
        finally:
            self.router.mark_done(self.node_id)
            self.heartbeat()
        return result

    async def astep(self, timeout: float | None = 0.1, max_messages: int = 32) -> bool:
        await self._ensure_started()

        if self.task_store is not None:
            claim_available = getattr(self.task_store, "claim_next_available", None)
            if not callable(claim_available):
                claimed = await self._claim_once()
                if claimed is not None:
                    self._metrics.claimed += 1
                    await self._process_claimed_record(claimed)
                    return True

        recv_task = asyncio.create_task(self.bus.recv(self.node_id, max_messages=max_messages, timeout=timeout))
        claim_task = None
        claim_available = getattr(self.task_store, "claim_next_available", None) if self.task_store else None
        if callable(claim_available):
            claim_task = asyncio.create_task(
                claim_available(self.node_id, sorted(self.capabilities), timeout=timeout)
            )

        tasks = [recv_task]
        if claim_task is not None:
            tasks.append(claim_task)

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        progressed = False

        if claim_task is not None and claim_task in done:
            claimed = claim_task.result()
            if claimed is not None:
                self._metrics.claimed += 1
                await self._process_claimed_record(claimed)
                progressed = True

        if recv_task in done:
            incoming = recv_task.result()
            if incoming:
                for envelope in incoming:
                    self._metrics.received += 1
                    await self._process_envelope(envelope)
                progressed = True

        if not progressed:
            self.heartbeat()
        return progressed

    async def run_forever(self, stop_event: asyncio.Event | None = None):
        await self._ensure_started()
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                await self.astep(timeout=None)
        finally:
            await self.close()

    async def _process_envelope(self, envelope):
        if not envelope.hop(self.node_id):
            await self._handle_expired(envelope)
            return

        if envelope.type == EnvelopeType.TASK:
            await self._handle_task(envelope)
        elif envelope.type == EnvelopeType.RESULT:
            await self._handle_result(envelope)

    async def _claim_once(self) -> TaskRecord | None:
        if self.task_store is None:
            return None
        await self.task_store.requeue_expired()
        for capability in sorted(self.capabilities):
            record = await self.task_store.claim_next(self.node_id, capability)
            if record is not None:
                return record
        return None

    async def _process_claimed_record(self, record: TaskRecord):
        task = record.task
        allowed, reason = self._can_execute_task(task)
        if not allowed:
            await self.task_store.fail(task.task_id, self.node_id, reason)
            self._metrics.failed += 1
            await self._checkpoint_mesh_task(task, "FAILED", record=record, error=reason)
            return

        if self.verbose:
            print(f"[{self.node_id}] claimed durable task {task.task_id} ({task.capability})")

        lease_task = self._start_lease_renewer(task)
        started_at = time.time()
        result = TaskResult(
            task_id=task.task_id,
            capability=task.capability,
            node_id=self.node_id,
            started_at=started_at,
            metadata={
                "attempts": record.attempts,
                "tenant_id": task.tenant_id,
                "memory_scope": task.memory_scope,
            },
        )

        self.router.mark_assigned(self.node_id)
        await self._checkpoint_mesh_task(task, "RUNNING", record=record)
        try:
            output = await self._execute_task(task)
            result.finalize(state=TaskState.COMPLETED, output=output)
            await self.task_store.complete(task.task_id, self.node_id, result)
            self._metrics.completed += 1
            await self._checkpoint_mesh_task(task, "COMPLETED", record=record, result=result)
        except Exception as exc:
            await self.task_store.fail(task.task_id, self.node_id, str(exc))
            self._metrics.failed += 1
            await self._checkpoint_mesh_task(task, "FAILED", record=record, error=str(exc))
        finally:
            lease_task.cancel()
            await asyncio.gather(lease_task, return_exceptions=True)
            self.router.mark_done(self.node_id)
            self.heartbeat()

    def _start_lease_renewer(self, task: MeshTask) -> asyncio.Task:
        async def _loop():
            interval = max(min(float(task.lease_seconds) / 2.0, 5.0), 0.2)
            while True:
                await asyncio.sleep(interval)
                try:
                    await self.task_store.renew_lease(task.task_id, self.node_id)
                except Exception:
                    logger.debug("Lease renewal failed for task %s", task.task_id, exc_info=True)

        return asyncio.create_task(_loop(), name=f"async-lease-renew-{task.task_id[:8]}")

    async def _handle_expired(self, envelope):
        if envelope.type != EnvelopeType.TASK:
            return

        task_id = envelope.task_id or "unknown"
        capability = envelope.capability or "unknown"
        requester = envelope.payload.get("requester", envelope.source)

        result = TaskResult(
            task_id=task_id,
            capability=capability,
            node_id=self.node_id,
            metadata={"trace": list(envelope.trace)},
        )
        result.finalize(
            state=TaskState.EXPIRED,
            error=f"Task expired after TTL hops. Trace: {envelope.trace}",
        )
        self._metrics.failed += 1
        await self._deliver_result(result, requester)

    async def _handle_task(self, envelope):
        capability = envelope.capability or ""
        task_id = envelope.task_id or ""
        requester = envelope.payload.get("requester", envelope.source)

        if capability not in self.capabilities:
            await self._forward_or_fail(envelope, f"Node lacks capability '{capability}'")
            return

        task = MeshTask(
            capability=capability,
            prompt=envelope.payload.get("prompt", ""),
            requester=requester,
            task_id=task_id,
            priority=envelope.priority,
            ttl=envelope.ttl,
            affinity=envelope.affinity,
            metadata=dict(envelope.payload.get("metadata", {})),
            task_policy=dict(envelope.payload.get("task_policy", {})),
            max_attempts=int(envelope.payload.get("max_attempts", 3)),
            lease_seconds=float(envelope.payload.get("lease_seconds", 30.0)),
            tenant_id=str(envelope.payload.get("tenant_id", "")),
            memory_scope=str(envelope.payload.get("memory_scope", "")),
            parent_task_id=envelope.payload.get("parent_task_id"),
            min_trust_tier=str(envelope.payload.get("min_trust_tier", "standard")),
        )

        allowed, reason = self._can_execute_task(task)
        if not allowed:
            result = TaskResult(
                task_id=task.task_id,
                capability=task.capability,
                node_id=self.node_id,
                metadata={"trace": list(envelope.trace)},
            )
            result.finalize(state=TaskState.FAILED, error=reason)
            self._metrics.failed += 1
            await self._deliver_result(result, requester)
            await self._record_audit("task_failed", task=task, message=reason)
            return

        started_at = time.time()
        result = TaskResult(
            task_id=task.task_id,
            capability=task.capability,
            node_id=self.node_id,
            started_at=started_at,
            metadata={"trace": list(envelope.trace)},
        )

        self.router.mark_assigned(self.node_id)
        await self._record_audit("task_claimed", task=task, message="task claimed")
        await self._checkpoint_mesh_task(task, "RUNNING")
        try:
            output = await self._execute_task(task)
            result.finalize(state=TaskState.COMPLETED, output=output)
            self._metrics.completed += 1
            await self._record_audit("task_completed", task=task, message="task completed")
            await self._checkpoint_mesh_task(task, "COMPLETED", result=result)
        except Exception as exc:
            result.finalize(state=TaskState.FAILED, error=str(exc))
            self._metrics.failed += 1
            await self._record_audit("task_failed", task=task, message=str(exc))
            await self._checkpoint_mesh_task(task, "FAILED", error=str(exc))
        finally:
            self.router.mark_done(self.node_id)
            self.heartbeat()

        await self._deliver_result(result, requester)

    async def _forward_or_fail(self, envelope, reason: str):
        visited = set(envelope.trace)
        visited.add(self.node_id)
        target = self.router.route(
            envelope.capability or "",
            exclude=visited,
            affinity=envelope.affinity,
            tenant_id=str(envelope.payload.get("tenant_id", "")),
            min_trust_tier=str(envelope.payload.get("min_trust_tier", "standard")),
            requester=envelope.payload.get("requester", envelope.source),
        )
        if target is None:
            requester = envelope.payload.get("requester", envelope.source)
            result = TaskResult(
                task_id=envelope.task_id or "unknown",
                capability=envelope.capability or "unknown",
                node_id=self.node_id,
                metadata={"trace": list(envelope.trace)},
            )
            result.finalize(state=TaskState.FAILED, error=f"{reason}. No alternate route found.")
            self._metrics.failed += 1
            await self._deliver_result(result, requester)
            return

        envelope.source = self.node_id
        envelope.destination = target
        delivered = await self.bus.send(envelope)
        if delivered > 0:
            self._metrics.forwarded += 1
        else:
            requester = envelope.payload.get("requester", envelope.source)
            result = TaskResult(
                task_id=envelope.task_id or "unknown",
                capability=envelope.capability or "unknown",
                node_id=self.node_id,
            )
            result.finalize(
                state=TaskState.FAILED,
                error=f"Routing selected '{target}' but message delivery failed.",
            )
            self._metrics.failed += 1
            await self._deliver_result(result, requester)

    async def _handle_result(self, envelope):
        if envelope.destination and envelope.destination != self.node_id:
            await self.bus.send(envelope)
            self._metrics.forwarded += 1
            return

        payload = envelope.payload
        raw_state = payload.get("state", TaskState.FAILED.value)
        try:
            state = TaskState(raw_state)
        except ValueError:
            state = TaskState.FAILED

        result = TaskResult(
            task_id=envelope.task_id or "unknown",
            capability=envelope.capability or "unknown",
            node_id=payload.get("node_id", envelope.source),
            state=state,
            output=payload.get("output", ""),
            error=payload.get("error"),
            started_at=payload.get("started_at", 0.0),
            finished_at=payload.get("finished_at"),
            metadata=dict(payload.get("metadata", {})),
        )
        self._results[result.task_id] = result
        self._result_event_for(result.task_id).set()

    async def _deliver_result(self, result: TaskResult, requester: str):
        if self.task_store is not None:
            if requester == self.node_id:
                self._results[result.task_id] = result
                self._result_event_for(result.task_id).set()
            return

        if requester == self.node_id:
            self._results[result.task_id] = result
            self._result_event_for(result.task_id).set()
            return

        envelope = make_result_envelope(result=result, source=self.node_id, destination=requester)
        delivered = await self.bus.send(envelope)
        if delivered == 0:
            self._results[result.task_id] = result
            self._result_event_for(result.task_id).set()

    async def _execute_task(self, task: MeshTask) -> str:
        task_policy = TaskPolicy.from_dict(task.task_policy)
        task_context = {
            "task_id": task.task_id,
            "capability": task.capability,
            "tenant_id": task.tenant_id,
            "parent_task_id": task.parent_task_id or "",
            "memory_scope": task.memory_scope,
            "requester": task.requester,
        }

        if self.task_handler is not None:
            if inspect.iscoroutinefunction(self.task_handler):
                return str(await self.task_handler(task))
            return str(await asyncio.to_thread(self.task_handler, task))

        if self.agent is not None:
            if self.agent.tools and len(self.agent.tools) > 0:
                result = await asyncio.to_thread(
                    self.agent.run,
                    task.prompt,
                    task_policy,
                    task_context,
                )
                return result.answer
            return await asyncio.to_thread(
                self.agent.chat,
                task.prompt,
                task_policy,
                task_context,
            )

        return f"[{self.node_id}] completed '{task.capability}' for prompt: {task.prompt}"

    def _can_execute_task(self, task: MeshTask) -> tuple[bool, str]:
        if task.capability not in self.capabilities:
            return False, f"Node lacks capability '{task.capability}'"

        manifest_tenant = self.manifest.tenant_id
        if task.tenant_id and manifest_tenant and task.tenant_id != manifest_tenant:
            return False, f"task tenant '{task.tenant_id}' does not match node tenant '{manifest_tenant}'"

        if not trust_tier_allows(self.manifest.trust_tier, task.min_trust_tier):
            return False, (
                f"node trust tier '{self.manifest.trust_tier}' "
                f"does not satisfy minimum '{task.min_trust_tier}'"
            )

        if self.manifest.delegation_allowlist and task.requester != self.node_id:
            if not any(fnmatch(task.requester, pattern) for pattern in self.manifest.delegation_allowlist):
                return False, f"requester '{task.requester}' is not allowed to delegate to this node"

        return True, "ok"

    async def _record_audit(
        self,
        event_type: str,
        *,
        task: MeshTask | None = None,
        message: str = "",
        metadata: dict[str, Any] | None = None,
        **extra: Any,
    ):
        if self.audit_sink is None:
            return
        record_fn = getattr(self.audit_sink, "record_event", None)
        if not callable(record_fn):
            return
        payload = {
            "event_type": event_type,
            "node_id": self.node_id,
            "message": message,
            "metadata": dict(metadata or {}),
            **extra,
        }
        if task is not None:
            payload.setdefault("task_id", task.task_id)
            payload.setdefault("tenant_id", task.tenant_id)
            payload.setdefault("parent_task_id", task.parent_task_id or "")

        result = record_fn(payload)
        if inspect.isawaitable(result):
            await result

    def _mesh_run_id(self, task: MeshTask) -> str:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        return str(metadata.get("jgx_run_id") or metadata.get("run_id") or f"mesh_{task.task_id}")

    async def _ensure_mesh_state_run(self, task: MeshTask) -> str | None:
        if self.state_store is None:
            return None
        run_id = self._mesh_run_id(task)
        if run_id in self._state_runs:
            return run_id
        manifest = JadeStateManifest(
            run_id=run_id,
            task_id=task.task_id,
            agent_id=self.node_id,
            tenant_id=task.tenant_id,
            capability=task.capability,
            state_kind="mesh_task",
            memory_scope_hash=canonical_json_hash({
                "tenant_id": task.tenant_id,
                "memory_scope": task.memory_scope,
            }),
            metadata={
                "requester": task.requester,
                "parent_task_id": task.parent_task_id or "",
                "min_trust_tier": task.min_trust_tier,
            },
        )
        await asyncio.to_thread(self.state_store.create_run, manifest)
        await asyncio.to_thread(self.state_store.append_event, run_id, JadeStateEvent(
            event_type="mesh_task_started",
            run_id=run_id,
            phase="NEW",
            actor=self.node_id,
            message="mesh task state run started",
        ))
        self._state_runs.add(run_id)
        return run_id

    async def _checkpoint_mesh_task(
        self,
        task: MeshTask,
        phase: str,
        *,
        record: TaskRecord | None = None,
        result: TaskResult | None = None,
        error: str = "",
    ) -> None:
        run_id = await self._ensure_mesh_state_run(task)
        if self.state_store is None or not run_id:
            return

        attempt = int(getattr(record, "attempts", 0) or 0)
        snapshot = AgentRuntimeSnapshot(
            phase=phase,
            step=attempt,
            mesh=MeshRuntimeSnapshot(
                shard_key=task.affinity or task.capability,
                lease_owner=self.node_id,
                lease_deadline=float(getattr(record, "lease_deadline", 0.0) or 0.0),
                attempt=attempt,
                task_state=phase.lower(),
                task_metadata={
                    "task_id": task.task_id,
                    "capability": task.capability,
                    "tenant_id": task.tenant_id,
                    "memory_scope": task.memory_scope,
                    "output": result.output if result is not None else "",
                    "error": error or (result.error if result is not None else ""),
                },
            ),
        )
        await asyncio.to_thread(self.state_store.save_snapshot, run_id, snapshot)
        await asyncio.to_thread(self.state_store.append_event, run_id, JadeStateEvent(
            event_type="checkpoint",
            run_id=run_id,
            phase=phase,
            step=attempt,
            actor=self.node_id,
            message="mesh task checkpoint saved",
            payload={"snapshot_id": snapshot.snapshot_id, "task_id": task.task_id},
        ))

    @staticmethod
    def _infer_affinity(task: MeshTask) -> str | None:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        command = metadata.get("command")
        if isinstance(command, dict):
            drone_id = command.get("drone_id")
            if drone_id:
                return f"drone:{drone_id}"
        return None

    def __repr__(self) -> str:
        caps = ",".join(sorted(self.capabilities))
        return f"<AsyncMeshNode({self.node_id}, caps=[{caps}])>"
