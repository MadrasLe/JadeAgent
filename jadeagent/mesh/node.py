"""
Mesh node runtime and in-memory transport.

This module provides:
- InMemoryMeshBus: local message transport for simulation and testing.
- MeshNode: autonomous worker that can execute tasks and forward messages.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass
from dataclasses import replace
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any, Callable

from .protocol import (
    EnvelopeType,
    MeshEnvelope,
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

logger = logging.getLogger("jadeagent.mesh")

TaskHandler = Callable[[MeshTask], str]
_LEGACY_RUNTIME_WARNED = False


@dataclass
class NodeMetrics:
    """Basic counters for mesh node behavior."""

    received: int = 0
    forwarded: int = 0
    completed: int = 0
    failed: int = 0
    claimed: int = 0


class InMemoryMeshBus:
    """
    In-process transport for mesh envelopes.

    Useful for local testing and simulation without sockets or brokers.
    """

    def __init__(self):
        self._nodes: dict[str, MeshNode] = {}

    def register(self, node: MeshNode):
        self._nodes[node.node_id] = node

    def unregister(self, node_id: str):
        self._nodes.pop(node_id, None)

    def send(self, envelope: MeshEnvelope) -> int:
        """
        Send an envelope to one node, or broadcast if destination is None.
        """
        delivered = 0
        if envelope.destination is None:
            for node_id, node in self._nodes.items():
                if node_id == envelope.source:
                    continue
                if node.enqueue(copy.deepcopy(envelope)):
                    delivered += 1
            return delivered

        node = self._nodes.get(envelope.destination)
        if node is None:
            return 0
        if node.enqueue(copy.deepcopy(envelope)):
            delivered += 1
        return delivered

    def run_until_idle(self, max_cycles: int = 2000) -> int:
        """
        Tick all nodes until no work remains or max_cycles is reached.
        """
        cycles = 0
        for _ in range(max_cycles):
            progressed = False
            for node in self._nodes.values():
                progressed = node.step() or progressed
            cycles += 1
            if not progressed and all(not n.has_pending_messages for n in self._nodes.values()):
                break
        return cycles

    def poll(self, node_id: str, max_messages: int = 32) -> list[MeshEnvelope]:
        """In-memory transport pushes directly, so poll is a no-op."""
        return []


class MeshNode:
    """
    Autonomous agent node in the mesh.

    A node can:
    - receive task envelopes
    - execute tasks if it has the required capability
    - claim durable tasks from a TaskStore
    - forward tasks to a better node when needed
    - collect result envelopes for submitted tasks
    """

    def __init__(
        self,
        node_id: str,
        capabilities: set[str] | list[str] | tuple[str, ...],
        router: MeshRouter,
        bus: MeshTransport,
        agent: Agent | None = None,
        task_handler: TaskHandler | None = None,
        max_inflight: int = 2,
        max_queue: int = 256,
        verbose: bool = False,
        manifest: NodeManifest | None = None,
        task_store: TaskStore | None = None,
        memory_router: MemoryRouter | None = None,
        audit_sink: Any = None,
        state_store: StateStore | None = None,
    ):
        self.node_id = node_id
        self.capabilities = set(capabilities)
        self.router = router
        self.bus = bus
        self.agent = agent
        self.task_handler = task_handler
        self.max_inflight = max_inflight
        self.max_queue = max_queue
        self.verbose = verbose
        self.task_store = task_store
        self.memory_router = memory_router
        self.audit_sink = audit_sink or task_store
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

        global _LEGACY_RUNTIME_WARNED
        if self.task_store is None and not _LEGACY_RUNTIME_WARNED:
            warnings.warn(
                (
                    "MeshNode is running in legacy in-memory task mode; "
                    "task delivery and results are not durable until a TaskStore is configured."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            _LEGACY_RUNTIME_WARNED = True

        self._inbox: deque[MeshEnvelope] = deque()
        self._seen_messages: set[str] = set()
        self._results: dict[str, TaskResult] = {}
        self._metrics = NodeMetrics()

        self.router.register_node(
            node_id=self.node_id,
            capabilities=self.capabilities,
            max_inflight=self.max_inflight,
            metadata=self.manifest.routing_metadata(),
        )
        self.bus.register(self)
        self.heartbeat()

    @property
    def has_pending_messages(self) -> bool:
        return bool(self._inbox)

    @property
    def queue_depth(self) -> int:
        return len(self._inbox)

    @property
    def metrics(self) -> dict[str, int]:
        return {
            "received": self._metrics.received,
            "forwarded": self._metrics.forwarded,
            "completed": self._metrics.completed,
            "failed": self._metrics.failed,
            "claimed": self._metrics.claimed,
        }

    def enqueue(self, envelope: MeshEnvelope) -> bool:
        """Queue an inbound envelope. Returns False if queue is full."""
        if len(self._inbox) >= self.max_queue:
            return False
        self._inbox.append(envelope)
        self.heartbeat()
        return True

    def heartbeat(self):
        """Publish liveness and queue depth to the router."""
        self.router.update_heartbeat(self.node_id, queue_depth=len(self._inbox))

    def submit_task(self, task: MeshTask) -> str:
        """
        Submit a new task into the mesh from this node.

        Returns:
            task_id
        """
        if task.requester == "client":
            task.requester = self.node_id

        if not task.tenant_id:
            task.tenant_id = self.manifest.tenant_id

        if task.affinity is None:
            task.affinity = self._infer_affinity(task)

        if self.task_store is not None:
            self.task_store.submit(task)
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
            self._metrics.failed += 1
            self._record_audit(
                "task_failed",
                task=task,
                message=result.error or "",
            )
            return task.task_id

        envelope.destination = target
        self.bus.send(envelope)
        self._record_audit("task_submitted", task=task, message="task submitted")
        return task.task_id

    def get_result(self, task_id: str) -> TaskResult | None:
        """Get a stored task result for tasks requested by this node."""
        if self.task_store is not None:
            record = self.task_store.get(task_id)
            if record is not None and record.state in {
                TaskState.COMPLETED,
                TaskState.FAILED,
                TaskState.CANCELLED,
            }:
                return record.to_result()
        return self._results.get(task_id)

    def step(self) -> bool:
        """
        Process one unit of work.

        Returns:
            True if any work was processed, False if idle.
        """
        self._drain_transport()

        if self._inbox:
            envelope = self._inbox.popleft()
            self.heartbeat()

            if envelope.message_id in self._seen_messages:
                return True
            self._seen_messages.add(envelope.message_id)
            self._metrics.received += 1

            if not envelope.hop(self.node_id):
                self._handle_expired(envelope)
                return True

            if envelope.type == EnvelopeType.TASK:
                self._handle_task(envelope)
            elif envelope.type == EnvelopeType.RESULT:
                self._handle_result(envelope)
            return True

        claimed = self._claim_from_store()
        if claimed:
            return True

        self.heartbeat()
        return False

    def _drain_transport(self, max_messages: int = 32):
        poll_fn = getattr(self.bus, "poll", None)
        if not callable(poll_fn):
            return

        try:
            incoming = poll_fn(self.node_id, max_messages=max_messages)
        except TypeError:
            incoming = poll_fn(self.node_id)
        except Exception:
            return

        if not incoming:
            return

        for envelope in incoming:
            if envelope.destination is None and envelope.source == self.node_id:
                continue
            self.enqueue(envelope)

    def _claim_from_store(self) -> bool:
        if self.task_store is None:
            return False

        try:
            self.task_store.requeue_expired()
        except Exception:
            logger.debug("Task store sweeper failed", exc_info=True)

        for capability in sorted(self.capabilities):
            record = self.task_store.claim_next(self.node_id, capability)
            if record is None:
                continue
            self._metrics.claimed += 1
            self._process_claimed_record(record)
            return True
        return False

    def _process_claimed_record(self, record: TaskRecord):
        task = record.task
        allowed, reason = self._can_execute_task(task)
        if not allowed:
            self.task_store.fail(task.task_id, self.node_id, reason)
            self._metrics.failed += 1
            self._checkpoint_mesh_task(task, "FAILED", record=record, error=reason)
            return

        if self.verbose:
            print(f"[{self.node_id}] claimed durable task {task.task_id} ({task.capability})")

        self._checkpoint_mesh_task(task, "RUNNING", record=record)
        stop_lease = self._start_lease_renewer(task)
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
        try:
            output = self._execute_task(task)
            result.finalize(state=TaskState.COMPLETED, output=output)
            self.task_store.complete(task.task_id, self.node_id, result)
            self._metrics.completed += 1
            self._checkpoint_mesh_task(task, "COMPLETED", record=record, result=result)
        except Exception as exc:
            self.task_store.fail(task.task_id, self.node_id, str(exc))
            self._metrics.failed += 1
            self._checkpoint_mesh_task(task, "FAILED", record=record, error=str(exc))
        finally:
            stop_lease()
            self.router.mark_done(self.node_id)
            self.heartbeat()

    def _start_lease_renewer(self, task: MeshTask) -> Callable[[], None]:
        if self.task_store is None:
            return lambda: None

        stop_event = threading.Event()
        interval = max(min(float(task.lease_seconds) / 2.0, 5.0), 0.2)

        def _loop():
            while not stop_event.wait(interval):
                try:
                    self.task_store.renew_lease(task.task_id, self.node_id)
                except Exception:
                    logger.debug("Lease renewal failed for task %s", task.task_id, exc_info=True)

        thread = threading.Thread(
            target=_loop,
            name=f"lease-renew-{task.task_id[:8]}",
            daemon=True,
        )
        thread.start()

        def _stop():
            stop_event.set()
            thread.join(timeout=0.2)

        return _stop

    def _handle_expired(self, envelope: MeshEnvelope):
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
        self._deliver_result(result, requester)

    def _handle_task(self, envelope: MeshEnvelope):
        capability = envelope.capability or ""
        task_id = envelope.task_id or ""
        requester = envelope.payload.get("requester", envelope.source)

        if capability not in self.capabilities:
            self._forward_or_fail(envelope, f"Node lacks capability '{capability}'")
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
            self._deliver_result(result, requester)
            self._record_audit("task_failed", task=task, message=reason)
            self._checkpoint_mesh_task(task, "FAILED", error=reason)
            return

        if self.verbose:
            print(f"[{self.node_id}] executing {task_id} ({capability})")

        started_at = time.time()
        result = TaskResult(
            task_id=task.task_id,
            capability=task.capability,
            node_id=self.node_id,
            started_at=started_at,
            metadata={"trace": list(envelope.trace)},
        )

        self.router.mark_assigned(self.node_id)
        self._record_audit("task_claimed", task=task, message="task claimed")
        self._checkpoint_mesh_task(task, "RUNNING")
        try:
            output = self._execute_task(task)
            result.finalize(state=TaskState.COMPLETED, output=output)
            self._metrics.completed += 1
            self._record_audit("task_completed", task=task, message="task completed")
            self._checkpoint_mesh_task(task, "COMPLETED", result=result)
        except Exception as exc:
            result.finalize(state=TaskState.FAILED, error=str(exc))
            self._metrics.failed += 1
            self._record_audit("task_failed", task=task, message=str(exc))
            self._checkpoint_mesh_task(task, "FAILED", error=str(exc))
        finally:
            self.router.mark_done(self.node_id)
            self.heartbeat()

        self._deliver_result(result, requester)

    def _forward_or_fail(self, envelope: MeshEnvelope, reason: str):
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
            result.finalize(
                state=TaskState.FAILED,
                error=f"{reason}. No alternate route found.",
            )
            self._metrics.failed += 1
            self._deliver_result(result, requester)
            return

        envelope.source = self.node_id
        envelope.destination = target
        delivered = self.bus.send(envelope)
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
            self._deliver_result(result, requester)

    def _handle_result(self, envelope: MeshEnvelope):
        if envelope.destination and envelope.destination != self.node_id:
            self.bus.send(envelope)
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
            started_at=payload.get("started_at", envelope.created_at),
            finished_at=payload.get("finished_at"),
            metadata=dict(payload.get("metadata", {})),
        )
        self._results[result.task_id] = result

    def _deliver_result(self, result: TaskResult, requester: str):
        if self.task_store is not None:
            if requester == self.node_id:
                self._results[result.task_id] = result
            return

        if requester == self.node_id:
            self._results[result.task_id] = result
            return

        envelope = make_result_envelope(
            result=result,
            source=self.node_id,
            destination=requester,
        )
        delivered = self.bus.send(envelope)
        if delivered == 0:
            self._results[result.task_id] = result

    def _execute_task(self, task: MeshTask) -> str:
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
            return str(self.task_handler(task))

        if self.agent is not None:
            if self.agent.tools and len(self.agent.tools) > 0:
                return self.agent.run(
                    task.prompt,
                    task_policy=task_policy,
                    task_context=task_context,
                ).answer
            return self.agent.chat(
                task.prompt,
                task_policy=task_policy,
                task_context=task_context,
            )

        return (
            f"[{self.node_id}] completed '{task.capability}' "
            f"for prompt: {task.prompt}"
        )

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

    def _record_audit(
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
        try:
            record_fn(payload)
        except Exception:
            logger.debug("Failed to record audit event", exc_info=True)

    def _mesh_run_id(self, task: MeshTask) -> str:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        return str(metadata.get("jgx_run_id") or metadata.get("run_id") or f"mesh_{task.task_id}")

    def _ensure_mesh_state_run(self, task: MeshTask) -> str | None:
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
        self.state_store.create_run(manifest)
        self.state_store.append_event(run_id, JadeStateEvent(
            event_type="mesh_task_started",
            run_id=run_id,
            phase="NEW",
            actor=self.node_id,
            message="mesh task state run started",
        ))
        self._state_runs.add(run_id)
        return run_id

    def _checkpoint_mesh_task(
        self,
        task: MeshTask,
        phase: str,
        *,
        record: TaskRecord | None = None,
        result: TaskResult | None = None,
        error: str = "",
    ) -> None:
        run_id = self._ensure_mesh_state_run(task)
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
        self.state_store.save_snapshot(run_id, snapshot)
        self.state_store.append_event(run_id, JadeStateEvent(
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
        return f"<MeshNode({self.node_id}, caps=[{caps}], q={self.queue_depth})>"
