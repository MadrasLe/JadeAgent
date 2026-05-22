"""Deterministic evaluation suites for JadeAgent governed execution."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import textwrap
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .backends.base import LLMBackend
from .core.agent import Agent
from .core.tools import tool
from .core.types import Message, Response, StreamChunk, ToolCall, Usage
from .mesh import InMemoryMeshBus, InMemoryTaskStore, MeshNode, MeshRouter, MeshTask
from .state.compatibility import validate_restore_compatibility
from .state.events import JadeStateEvent
from .state.integrity import verify_capsule
from .state.manifest import JadeStateManifest
from .state.snapshot import AgentRuntimeSnapshot
from .state.store import StateStore


class ScriptedBackend(LLMBackend):
    """Small deterministic backend used by local eval suites."""

    def __init__(self, responses: Iterable[Response] | None = None, *, model: str = "scripted-eval"):
        self._responses = list(responses or [])
        self.model = model

    def chat(
        self,
        messages: list[Message],
        tools=None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop=None,
    ) -> Response:
        if self._responses:
            response = self._responses.pop(0)
        else:
            response = Response(content="done")
        if response.model is None:
            response.model = self.model
        if response.finish_reason is None:
            response.finish_reason = "stop"
        if response.usage is None:
            response.usage = estimate_usage(messages, response, tools=tools)
        return response

    def stream(
        self,
        messages: list[Message],
        tools=None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop=None,
    ):
        if False:
            yield StreamChunk()
        return


@dataclass
class EvalCaseResult:
    """Result for one concrete eval case execution."""

    name: str
    run_id: str
    success: bool
    duration_ms: float
    run_ids: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str = ""
    repetition: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "run_id": self.run_id,
            "run_ids": list(self.run_ids or [self.run_id]),
            "success": self.success,
            "duration_ms": round(float(self.duration_ms), 3),
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "error": self.error,
            "repetition": self.repetition,
        }


EvalCase = Callable[[StateStore, str, int, Path, str], EvalCaseResult]


def _estimate_tokens(value: Any) -> int:
    text = json.dumps(value, ensure_ascii=True, sort_keys=True) if not isinstance(value, str) else value
    text = text or ""
    return max(1, int((len(text) + 3) / 4))


def estimate_usage(messages: list[Message], response: Response, *, tools: Any = None) -> Usage:
    prompt_payload = [message.to_dict() for message in messages]
    if tools:
        prompt_payload.append({"tools": [tool_schema.to_dict() for tool_schema in tools]})
    completion_payload: Any = response.content or ""
    if response.tool_calls:
        completion_payload = [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in response.tool_calls
        ]
    prompt_tokens = _estimate_tokens(prompt_payload)
    completion_tokens = _estimate_tokens(completion_payload)
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def usage_metrics(usage: Usage | None, *, prefix: str = "", estimated: bool = True) -> dict[str, Any]:
    if usage is None:
        return {
            f"{prefix}prompt_tokens": 0,
            f"{prefix}completion_tokens": 0,
            f"{prefix}total_tokens": 0,
            f"{prefix}usage_estimated": estimated,
            f"{prefix}cost_estimate_usd": 0.0,
        }
    return {
        f"{prefix}prompt_tokens": int(usage.prompt_tokens),
        f"{prefix}completion_tokens": int(usage.completion_tokens),
        f"{prefix}total_tokens": int(usage.total_tokens),
        f"{prefix}usage_estimated": estimated,
        f"{prefix}cost_estimate_usd": 0.0,
    }


def merge_usage_metrics(usages: Iterable[Usage | None], *, prefix: str = "", estimated: bool = True) -> dict[str, Any]:
    totals = Usage()
    for usage in usages:
        if usage is None:
            continue
        totals.prompt_tokens += int(usage.prompt_tokens)
        totals.completion_tokens += int(usage.completion_tokens)
        totals.total_tokens += int(usage.total_tokens)
    return usage_metrics(totals, prefix=prefix, estimated=estimated)


def _now_token() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    return path


def _safe_child(root: Path, relative: str) -> Path:
    root = root.resolve()
    child = (root / relative).resolve()
    try:
        child.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes eval output root: {relative}") from exc
    return child


def _tag_run(
    store: StateStore,
    run_id: str,
    *,
    suite: str,
    case_name: str,
    repetition: int,
    success: bool,
    role: str = "primary",
    extra: dict[str, Any] | None = None,
) -> None:
    try:
        manifest = store.load_run(run_id).manifest
    except Exception:
        manifest = JadeStateManifest(run_id=run_id)
    manifest.metadata.update({
        "eval_suite": suite,
        "eval_case": case_name,
        "eval_repetition": repetition,
        "eval_success": bool(success),
        "eval_case_role": role,
    })
    if extra:
        manifest.metadata.update(extra)
    store.create_run(manifest)


def _complete_manual_run(
    store: StateStore,
    run_id: str,
    *,
    case_name: str,
    suite: str,
    repetition: int,
    success: bool,
    metrics: dict[str, Any] | None = None,
    duration_ms: float | None = None,
) -> None:
    phase = "COMPLETED" if success else "FAILED"
    manifest = JadeStateManifest(
        run_id=run_id,
        agent_id="jade-eval",
        capability=case_name,
        state_kind="eval_case",
        metadata={
            "eval_suite": suite,
            "eval_case": case_name,
            "eval_repetition": repetition,
            "eval_success": bool(success),
            "eval_case_role": "primary",
            "eval_metrics": dict(metrics or {}),
            "eval_duration_ms": duration_ms,
        },
    )
    store.create_run(manifest)
    store.append_event(run_id, JadeStateEvent(
        event_type="eval_case_completed" if success else "eval_case_failed",
        run_id=run_id,
        phase=phase,
        actor="jade-eval",
        message=f"eval case {case_name} {'passed' if success else 'failed'}",
        payload=dict(metrics or {}),
    ))
    store.save_snapshot(run_id, AgentRuntimeSnapshot(
        phase=phase,
        step=1,
        metadata={"eval_case": case_name, "metrics": dict(metrics or {})},
    ))


def capsule_metrics(store: StateStore, run_id: str) -> dict[str, Any]:
    """Compute stable presentation metrics for one stored JGX run."""

    capsule = store.load_run(run_id)
    latest = capsule.latest_snapshot
    event_counts = Counter(event.event_type for event in capsule.events)
    phase_counts = Counter(snapshot.phase for snapshot in capsule.snapshots)
    timestamps = [event.timestamp for event in capsule.events]
    timestamps.extend(snapshot.created_at for snapshot in capsule.snapshots)
    duration_ms = 0.0
    if timestamps:
        duration_ms = max(0.0, (max(timestamps) - min(timestamps)) * 1000.0)

    idempotency_counts: Counter[str] = Counter()
    for event in capsule.events:
        if event.event_type != "tool_result_recorded":
            continue
        key = str((event.payload or {}).get("idempotency_key", ""))
        if key:
            idempotency_counts[key] += 1
    duplicate_tool_executions = sum(max(0, count - 1) for count in idempotency_counts.values())

    integrity = verify_capsule(capsule)
    recovery_success = (
        event_counts.get("simulated_crash", 0) > 0
        and latest is not None
        and latest.phase == "COMPLETED"
    )
    return {
        "run_id": run_id,
        "latest_phase": latest.phase if latest is not None else "",
        "success": bool(latest is not None and latest.phase == "COMPLETED"),
        "task_status": (latest.phase.lower() if latest is not None else "missing"),
        "task_completed": bool(latest is not None and latest.phase == "COMPLETED"),
        "duration_ms": round(duration_ms, 3),
        "event_count": len(capsule.events),
        "snapshot_count": len(capsule.snapshots),
        "checkpoint_count": event_counts.get("checkpoint", 0),
        "tool_results_recorded": event_counts.get("tool_result_recorded", 0),
        "tool_results_reused": event_counts.get("tool_result_reused", 0),
        "duplicate_tool_executions": duplicate_tool_executions,
        "recovery_success": recovery_success,
        "audit_complete": bool(capsule.manifest.run_id and capsule.events and capsule.snapshots),
        "event_counts": dict(sorted(event_counts.items())),
        "phase_counts": dict(sorted(phase_counts.items())),
        "integrity_ok": bool(integrity["ok"]),
        "secret_leak_count": int(integrity["secret_leak_count"]),
        "event_chain_hash": integrity["event_chain_hash"],
    }


def _case_state_restore(
    store: StateStore,
    suite: str,
    repetition: int,
    output_dir: Path,
    token: str,
) -> EvalCaseResult:
    case_name = "state_restore"
    run_id = f"eval_{case_name}_{token}_{repetition}"
    started = time.perf_counter()
    backend = ScriptedBackend([Response(content="state restore completed")])
    agent = Agent(
        backend=backend,
        name="eval_state_agent",
        state_store=store,
        run_id=run_id,
        verbose=False,
        max_iterations=1,
    )
    result = agent.run("Create a restorable state checkpoint.")

    restore_started = time.perf_counter()
    restored = Agent(
        backend=ScriptedBackend(),
        name="eval_state_agent",
        state_store=store,
        verbose=False,
    )
    snapshot = restored.restore_state(run_id)
    restore_latency_ms = (time.perf_counter() - restore_started) * 1000.0
    metrics = capsule_metrics(store, run_id)
    metrics.update({
        "restore_latency_ms": round(restore_latency_ms, 3),
        "restored_message_count": len(snapshot.messages),
        **usage_metrics(result.usage),
    })
    success = (
        result.answer == "state restore completed"
        and snapshot.phase == "COMPLETED"
        and len(restored.session.messages) >= 3
    )
    duration_ms = (time.perf_counter() - started) * 1000.0
    _tag_run(
        store,
        run_id,
        suite=suite,
        case_name=case_name,
        repetition=repetition,
        success=success,
        extra={"eval_metrics": metrics, "eval_duration_ms": duration_ms},
    )
    return EvalCaseResult(
        name=case_name,
        run_id=run_id,
        run_ids=[run_id],
        success=success,
        duration_ms=duration_ms,
        metrics=metrics,
        repetition=repetition,
    )


def _case_tool_idempotency(
    store: StateStore,
    suite: str,
    repetition: int,
    output_dir: Path,
    token: str,
) -> EvalCaseResult:
    case_name = "tool_idempotency"
    run_id = f"eval_{case_name}_{token}_{repetition}"
    started = time.perf_counter()
    calls = {"count": 0}

    @tool(description="Tool with a visible side effect")
    def eval_side_effect_tool(value: str) -> str:
        calls["count"] += 1
        return f"side-effect:{value}:{calls['count']}"

    def backend() -> ScriptedBackend:
        return ScriptedBackend([
            Response(tool_calls=[
                ToolCall(
                    id="stable_eval_call",
                    name="eval_side_effect_tool",
                    arguments={"value": "x"},
                )
            ]),
            Response(content="done"),
        ])

    usages: list[Usage | None] = []
    for _ in range(2):
        agent = Agent(
            backend=backend(),
            name="eval_idempotency_agent",
            tools=[eval_side_effect_tool],
            state_store=store,
            run_id=run_id,
            verbose=False,
            max_iterations=1,
        )
        result = agent.run("Call the side-effect tool exactly once.")
        usages.append(result.usage)

    metrics = capsule_metrics(store, run_id)
    metrics["side_effect_calls"] = calls["count"]
    metrics.update(merge_usage_metrics(usages))
    success = (
        calls["count"] == 1
        and metrics["tool_results_recorded"] == 1
        and metrics["tool_results_reused"] >= 1
        and metrics["duplicate_tool_executions"] == 0
    )
    duration_ms = (time.perf_counter() - started) * 1000.0
    _tag_run(
        store,
        run_id,
        suite=suite,
        case_name=case_name,
        repetition=repetition,
        success=success,
        extra={"eval_metrics": metrics, "eval_duration_ms": duration_ms},
    )
    return EvalCaseResult(
        name=case_name,
        run_id=run_id,
        run_ids=[run_id],
        success=success,
        duration_ms=duration_ms,
        metrics=metrics,
        repetition=repetition,
    )


def _case_crash_recovery(
    store: StateStore,
    suite: str,
    repetition: int,
    output_dir: Path,
    token: str,
) -> EvalCaseResult:
    case_name = "crash_recovery"
    run_id = f"eval_{case_name}_{token}_{repetition}"
    started = time.perf_counter()
    artifact = output_dir / f"{run_id}.txt"
    router = MeshRouter()
    bus = InMemoryMeshBus()
    task_store = InMemoryTaskStore()
    attempts: list[str] = []

    def latest_stage(stage: str) -> AgentRuntimeSnapshot | None:
        try:
            snapshots = store.load_run(run_id).snapshots
        except Exception:
            return None
        for snapshot in reversed(snapshots):
            if snapshot.metadata.get("stage") == stage:
                return snapshot
        return None

    def handler(task: MeshTask) -> str:
        worker = str(task.metadata.get("worker_name", "worker"))
        draft = latest_stage("draft_written")
        if draft is None:
            artifact.write_text("phase 1: draft persisted before crash\n", encoding="utf-8")
            snapshot = AgentRuntimeSnapshot(
                phase="CHECKPOINTING",
                step=1,
                metadata={"stage": "draft_written", "worker": worker, "artifact": str(artifact)},
            )
            store.save_snapshot(run_id, snapshot)
            store.append_event(run_id, JadeStateEvent(
                event_type="simulated_crash",
                run_id=run_id,
                phase="FAILED",
                step=1,
                actor=worker,
                message="worker crashed after durable checkpoint",
                payload={"snapshot_id": snapshot.snapshot_id},
            ))
            attempts.append(f"{worker}:crashed_after_checkpoint")
            raise RuntimeError("simulated crash after checkpoint")

        with artifact.open("a", encoding="utf-8") as handle:
            handle.write("phase 2: recovered worker completed artifact\n")
        snapshot = AgentRuntimeSnapshot(
            phase="COMPLETED",
            step=2,
            metadata={
                "stage": "completed_after_recovery",
                "worker": worker,
                "resumed_from_snapshot": draft.snapshot_id,
            },
        )
        store.save_snapshot(run_id, snapshot)
        store.append_event(run_id, JadeStateEvent(
            event_type="recovered_and_completed",
            run_id=run_id,
            phase="COMPLETED",
            step=2,
            actor=worker,
            message="worker resumed from checkpoint and completed",
            payload={"snapshot_id": snapshot.snapshot_id},
        ))
        attempts.append(f"{worker}:resumed_and_completed")
        return "recovered"

    def with_worker(task: MeshTask, worker_name: str) -> MeshTask:
        task.metadata = dict(task.metadata)
        task.metadata["worker_name"] = worker_name
        return task

    coordinator = MeshNode(
        node_id=f"eval_crash_coordinator_{repetition}",
        capabilities={"coordinate"},
        router=router,
        bus=bus,
        task_store=task_store,
        state_store=store,
        task_handler=lambda task: "coordinator",
    )
    worker_a = MeshNode(
        node_id=f"eval_crash_a_{repetition}",
        capabilities={"recover_eval_artifact"},
        router=router,
        bus=bus,
        task_store=task_store,
        state_store=store,
        task_handler=lambda task: handler(with_worker(task, "worker_before_crash")),
    )
    worker_b = MeshNode(
        node_id=f"eval_crash_b_{repetition}",
        capabilities={"recover_eval_artifact"},
        router=router,
        bus=bus,
        task_store=task_store,
        state_store=store,
        task_handler=lambda task: handler(with_worker(task, "worker_after_recovery")),
    )
    task = MeshTask(
        capability="recover_eval_artifact",
        prompt="Persist an artifact, crash, then recover.",
        requester=coordinator.node_id,
        metadata={"jgx_run_id": run_id},
        max_attempts=2,
        lease_seconds=1.0,
    )
    task_id = coordinator.submit_task(task)
    worker_a.step()
    worker_b.step()
    result = coordinator.get_result(task_id)
    record = task_store.get(task_id)

    metrics = capsule_metrics(store, run_id)
    metrics.update({
        "attempts": list(attempts),
        "final_task_state": record.state.value if record else "",
    })
    success = (
        result is not None
        and result.success
        and record is not None
        and record.state.value == "completed"
        and artifact.exists()
        and "phase 2" in artifact.read_text(encoding="utf-8")
        and metrics["recovery_success"]
    )
    duration_ms = (time.perf_counter() - started) * 1000.0
    _tag_run(
        store,
        run_id,
        suite=suite,
        case_name=case_name,
        repetition=repetition,
        success=success,
        extra={"artifact": str(artifact), "eval_metrics": metrics, "eval_duration_ms": duration_ms},
    )
    return EvalCaseResult(
        name=case_name,
        run_id=run_id,
        run_ids=[run_id],
        success=success,
        duration_ms=duration_ms,
        metrics=metrics,
        artifacts={"artifact": str(artifact)},
        repetition=repetition,
    )


def _case_compatibility_guard(
    store: StateStore,
    suite: str,
    repetition: int,
    output_dir: Path,
    token: str,
) -> EvalCaseResult:
    case_name = "compatibility_guard"
    run_id = f"eval_{case_name}_{token}_{repetition}"
    started = time.perf_counter()
    manifest = JadeStateManifest(
        run_id=run_id,
        agent_id="eval_compat_agent",
        tenant_id="tenant_a",
        capability="restore_guard",
        backend="ScriptedBackend",
        model_fingerprint="model-a",
        policy_hash="policy-a",
        tool_registry_hash="tools-a",
        metadata={
            "eval_suite": suite,
            "eval_case": case_name,
            "eval_repetition": repetition,
            "eval_case_role": "primary",
        },
    )
    store.create_run(manifest)
    report = validate_restore_compatibility(
        manifest,
        tenant_id="tenant_b",
        policy_hash="policy-a",
        tool_registry_hash="tools-a",
        model_fingerprint="model-a",
        backend="ScriptedBackend",
    )
    success = not report.allowed and any("tenant_id mismatch" in issue for issue in report.issues)
    metrics = {
        "blocked_restore_count": 0 if report.allowed else 1,
        "compatibility_allowed": report.allowed,
        "compatibility_issues": list(report.issues),
    }
    duration_ms = (time.perf_counter() - started) * 1000.0
    _complete_manual_run(
        store,
        run_id,
        case_name=case_name,
        suite=suite,
        repetition=repetition,
        success=success,
        metrics=metrics,
        duration_ms=duration_ms,
    )
    return EvalCaseResult(
        name=case_name,
        run_id=run_id,
        run_ids=[run_id],
        success=success,
        duration_ms=duration_ms,
        metrics={**capsule_metrics(store, run_id), **metrics},
        repetition=repetition,
    )


def _case_raw_call_baseline(
    store: StateStore,
    suite: str,
    repetition: int,
    output_dir: Path,
    token: str,
) -> EvalCaseResult:
    case_name = "raw_call_baseline"
    run_id = f"eval_{case_name}_{token}_{repetition}"
    task = "Return the baseline answer."
    started = time.perf_counter()

    raw_backend = ScriptedBackend([Response(content="baseline answer")])
    raw_started = time.perf_counter()
    raw_response = raw_backend.chat([
        Message.system("You are a baseline assistant."),
        Message.user(task),
    ])
    raw_runtime_ms = (time.perf_counter() - raw_started) * 1000.0

    jade_backend = ScriptedBackend([Response(content="baseline answer")])
    jade_started = time.perf_counter()
    agent = Agent(
        backend=jade_backend,
        name="eval_raw_baseline_agent",
        system_prompt="You are a baseline assistant.",
        state_store=store,
        run_id=run_id,
        verbose=False,
        max_iterations=1,
    )
    result = agent.run(task)
    jade_runtime_ms = (time.perf_counter() - jade_started) * 1000.0

    metrics = capsule_metrics(store, run_id)
    raw_usage = usage_metrics(raw_response.usage, prefix="raw_")
    jade_usage = usage_metrics(result.usage, prefix="jade_")
    raw_total = int(raw_usage["raw_total_tokens"])
    jade_total = int(jade_usage["jade_total_tokens"])
    metrics.update({
        **raw_usage,
        **jade_usage,
        "prompt_tokens": jade_usage["jade_prompt_tokens"],
        "completion_tokens": jade_usage["jade_completion_tokens"],
        "total_tokens": jade_usage["jade_total_tokens"],
        "raw_runtime_ms": round(raw_runtime_ms, 3),
        "jade_runtime_ms": round(jade_runtime_ms, 3),
        "state_overhead_ms": round(max(0.0, jade_runtime_ms - raw_runtime_ms), 3),
        "runtime_overhead_ratio": round(jade_runtime_ms / raw_runtime_ms, 3) if raw_runtime_ms > 0 else 0.0,
        "token_delta_vs_raw": jade_total - raw_total,
        "comparison_target": "raw_scripted_backend_chat",
    })
    success = (
        raw_response.content == result.answer
        and result.answer == "baseline answer"
        and metrics["task_completed"]
    )
    duration_ms = (time.perf_counter() - started) * 1000.0
    _tag_run(
        store,
        run_id,
        suite=suite,
        case_name=case_name,
        repetition=repetition,
        success=success,
        extra={"eval_metrics": metrics, "eval_duration_ms": duration_ms},
    )
    return EvalCaseResult(
        name=case_name,
        run_id=run_id,
        run_ids=[run_id],
        success=success,
        duration_ms=duration_ms,
        metrics=metrics,
        repetition=repetition,
    )


def _case_mesh_project(
    store: StateStore,
    suite: str,
    repetition: int,
    output_dir: Path,
    token: str,
) -> EvalCaseResult:
    case_name = "mesh_project"
    run_id = f"eval_{case_name}_{token}_{repetition}"
    started = time.perf_counter()
    project_root = output_dir / run_id
    router = MeshRouter()
    bus = InMemoryMeshBus()
    task_store = InMemoryTaskStore()
    run_ids: list[str] = []

    def write_models(task: MeshTask) -> str:
        _write_text(_safe_child(project_root, "taskpack/models.py"), """
            from __future__ import annotations

            from dataclasses import dataclass, field


            @dataclass(frozen=True)
            class Task:
                id: str
                title: str
                done: bool = False
                depends_on: tuple[str, ...] = field(default_factory=tuple)

                def complete(self, completed: set[str]) -> "Task":
                    missing = [dep for dep in self.depends_on if dep not in completed]
                    if missing:
                        raise ValueError(f"missing dependencies: {missing}")
                    return Task(self.id, self.title, True, self.depends_on)
        """)
        _write_text(_safe_child(project_root, "taskpack/__init__.py"), """
            from .models import Task
            from .planner import Planner

            __all__ = ["Planner", "Task"]
        """)
        return "models"

    def write_planner(task: MeshTask) -> str:
        _write_text(_safe_child(project_root, "taskpack/planner.py"), """
            from __future__ import annotations

            from .models import Task


            class Planner:
                def __init__(self) -> None:
                    self.tasks: dict[str, Task] = {}

                def add(self, task: Task) -> None:
                    if task.id in self.tasks:
                        raise ValueError(f"duplicate task: {task.id}")
                    missing = [dep for dep in task.depends_on if dep not in self.tasks]
                    if missing:
                        raise ValueError(f"unknown dependencies: {missing}")
                    self.tasks[task.id] = task

                def ready(self) -> list[str]:
                    completed = {task.id for task in self.tasks.values() if task.done}
                    return [
                        task.id
                        for task in self.tasks.values()
                        if not task.done and all(dep in completed for dep in task.depends_on)
                    ]

                def complete(self, task_id: str) -> None:
                    completed = {task.id for task in self.tasks.values() if task.done}
                    self.tasks[task_id] = self.tasks[task_id].complete(completed)

                def summary(self) -> dict[str, int]:
                    done = sum(1 for task in self.tasks.values() if task.done)
                    return {"total": len(self.tasks), "done": done, "open": len(self.tasks) - done}
        """)
        return "planner"

    def write_tests(task: MeshTask) -> str:
        _write_text(_safe_child(project_root, "tests/test_taskpack.py"), """
            import unittest

            from taskpack import Planner, Task


            class PlannerTests(unittest.TestCase):
                def test_dependency_readiness(self):
                    planner = Planner()
                    planner.add(Task("design", "Design API"))
                    planner.add(Task("build", "Build API", depends_on=("design",)))
                    self.assertEqual(planner.ready(), ["design"])
                    planner.complete("design")
                    self.assertEqual(planner.ready(), ["build"])

                def test_summary(self):
                    planner = Planner()
                    planner.add(Task("docs", "Write docs"))
                    self.assertEqual(planner.summary(), {"total": 1, "done": 0, "open": 1})


            if __name__ == "__main__":
                unittest.main()
        """)
        _write_text(_safe_child(project_root, "README.md"), """
            # taskpack

            Tiny deterministic project generated by a JadeAgent eval mesh.
        """)
        return "tests"

    def run_tests(task: MeshTask) -> str:
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
            cwd=project_root,
            text=True,
            capture_output=True,
            timeout=30,
        )
        _write_text(_safe_child(project_root, "test_result.json"), json.dumps({
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }, indent=2, ensure_ascii=True))
        if completed.returncode != 0:
            raise RuntimeError((completed.stdout + completed.stderr).strip())
        return "tests_passed"

    coordinator = MeshNode(
        node_id=f"eval_project_coordinator_{repetition}",
        capabilities={"coordinate"},
        router=router,
        bus=bus,
        task_store=task_store,
        state_store=store,
        task_handler=lambda task: "coordinator",
    )
    worker_specs = [
        ("eval_project_models", "write_models", write_models),
        ("eval_project_planner", "write_planner", write_planner),
        ("eval_project_tests", "write_tests", write_tests),
        ("eval_project_runner", "run_tests", run_tests),
    ]
    workers = [
        MeshNode(
            node_id=f"{node_id}_{repetition}",
            capabilities={capability},
            router=router,
            bus=bus,
            task_store=task_store,
            state_store=store,
            task_handler=handler,
        )
        for node_id, capability, handler in worker_specs
    ]

    for capability, _node_id, _handler in [(cap, node, handler) for node, cap, handler in worker_specs]:
        child_run_id = f"{run_id}_{capability}"
        run_ids.append(child_run_id)
        task = MeshTask(
            capability=capability,
            prompt=capability,
            requester=coordinator.node_id,
            metadata={"jgx_run_id": child_run_id, "project_root": str(project_root)},
            max_attempts=1,
        )
        task_id = coordinator.submit_task(task)
        bus.run_until_idle()
        result = coordinator.get_result(task_id)
        if result is None or not result.success:
            raise RuntimeError(result.error if result is not None else f"no result for {capability}")
        _tag_run(
            store,
            child_run_id,
            suite=suite,
            case_name=case_name,
            repetition=repetition,
            success=True,
            role="child",
            extra={"parent_eval_run_id": run_id},
        )

    files = [
        path
        for path in project_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    test_payload = json.loads((project_root / "test_result.json").read_text(encoding="utf-8"))
    success = test_payload["returncode"] == 0 and len(files) >= 5
    metrics = {
        "generated_files": len(files),
        "test_returncode": test_payload["returncode"],
        "unit_tests_passed": success,
        "mesh_child_runs": len(run_ids),
    }
    duration_ms = (time.perf_counter() - started) * 1000.0
    _complete_manual_run(
        store,
        run_id,
        case_name=case_name,
        suite=suite,
        repetition=repetition,
        success=success,
        metrics=metrics,
        duration_ms=duration_ms,
    )
    for child_run_id in run_ids:
        child_metrics = capsule_metrics(store, child_run_id)
        metrics.setdefault("child_event_count", 0)
        metrics.setdefault("child_snapshot_count", 0)
        metrics["child_event_count"] += child_metrics["event_count"]
        metrics["child_snapshot_count"] += child_metrics["snapshot_count"]
    return EvalCaseResult(
        name=case_name,
        run_id=run_id,
        run_ids=[run_id, *run_ids],
        success=success,
        duration_ms=duration_ms,
        metrics={**capsule_metrics(store, run_id), **metrics},
        artifacts={"project_root": str(project_root), "test_result": str(project_root / "test_result.json")},
        repetition=repetition,
    )


SUITES: dict[str, list[EvalCase]] = {
    "fast": [_case_state_restore, _case_tool_idempotency],
    "reliability": [
        _case_state_restore,
        _case_tool_idempotency,
        _case_crash_recovery,
        _case_compatibility_guard,
    ],
    "core": [
        _case_state_restore,
        _case_tool_idempotency,
        _case_crash_recovery,
        _case_compatibility_guard,
        _case_raw_call_baseline,
        _case_mesh_project,
    ],
}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return float(ordered[index])


def aggregate_results(results: list[EvalCaseResult]) -> dict[str, Any]:
    durations = [float(result.duration_ms) for result in results]
    successes = [result for result in results if result.success]
    completed = [result for result in results if result.metrics.get("task_completed") or result.success]
    recovery_cases = [result for result in results if result.name == "crash_recovery"]
    raw_baselines = [result for result in results if result.name == "raw_call_baseline"]
    restore_latencies = [
        float(result.metrics["restore_latency_ms"])
        for result in results
        if "restore_latency_ms" in result.metrics
    ]
    return {
        "case_runs": len(results),
        "passed": len(successes),
        "failed": len(results) - len(successes),
        "success_rate": round((len(successes) / len(results)) if results else 0.0, 4),
        "task_completion_rate": round((len(completed) / len(results)) if results else 0.0, 4),
        "recovery_success_rate": round(
            (
                sum(1 for result in recovery_cases if result.metrics.get("recovery_success"))
                / len(recovery_cases)
            )
            if recovery_cases else 0.0,
            4,
        ),
        "duplicate_tool_executions": sum(
            int(result.metrics.get("duplicate_tool_executions", 0))
            for result in results
        ),
        "secret_leak_count": sum(
            int(result.metrics.get("secret_leak_count", 0))
            for result in results
        ),
        "prompt_tokens": sum(int(result.metrics.get("prompt_tokens", 0)) for result in results),
        "completion_tokens": sum(int(result.metrics.get("completion_tokens", 0)) for result in results),
        "total_tokens": sum(int(result.metrics.get("total_tokens", 0)) for result in results),
        "cost_estimate_usd": round(sum(
            float(result.metrics.get("cost_estimate_usd", 0.0) or 0.0)
            for result in results
        ), 6),
        "raw_baseline_count": len(raw_baselines),
        "raw_runtime_ms_p50": round(percentile([
            float(result.metrics.get("raw_runtime_ms", 0.0))
            for result in raw_baselines
        ], 0.50), 3),
        "jade_runtime_ms_p50": round(percentile([
            float(result.metrics.get("jade_runtime_ms", 0.0))
            for result in raw_baselines
        ], 0.50), 3),
        "state_overhead_ms_p50": round(percentile([
            float(result.metrics.get("state_overhead_ms", 0.0))
            for result in raw_baselines
        ], 0.50), 3),
        "avg_events_per_run": round(statistics.mean([
            int(result.metrics.get("event_count", 0))
            for result in results
        ]) if results else 0.0, 3),
        "avg_snapshots_per_run": round(statistics.mean([
            int(result.metrics.get("snapshot_count", 0))
            for result in results
        ]) if results else 0.0, 3),
        "runtime_ms_p50": round(percentile(durations, 0.50), 3),
        "runtime_ms_p95": round(percentile(durations, 0.95), 3),
        "restore_latency_ms_p50": round(percentile(restore_latencies, 0.50), 3),
        "restore_latency_ms_p95": round(percentile(restore_latencies, 0.95), 3),
    }


def run_eval_suite(
    store: StateStore,
    *,
    suite: str = "core",
    runs: int = 1,
    output_dir: str | Path = "examples/generated/eval",
) -> dict[str, Any]:
    """Run a deterministic eval suite and return serializable results."""

    if suite not in SUITES:
        raise ValueError(f"unknown eval suite {suite!r}; available: {', '.join(sorted(SUITES))}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    token = _now_token()
    started = time.time()
    results: list[EvalCaseResult] = []
    for repetition in range(1, max(int(runs), 1) + 1):
        for case in SUITES[suite]:
            case_started = time.perf_counter()
            try:
                results.append(case(store, suite, repetition, output_path, token))
            except Exception as exc:
                case_name = case.__name__.removeprefix("_case_")
                failed_run_id = f"eval_{case_name}_{token}_{repetition}_failed"
                metrics = {"error_type": exc.__class__.__name__, "error": str(exc)}
                _complete_manual_run(
                    store,
                    failed_run_id,
                    case_name=case_name,
                    suite=suite,
                    repetition=repetition,
                    success=False,
                    metrics=metrics,
                )
                results.append(EvalCaseResult(
                    name=case_name,
                    run_id=failed_run_id,
                    run_ids=[failed_run_id],
                    success=False,
                    duration_ms=(time.perf_counter() - case_started) * 1000.0,
                    metrics={**capsule_metrics(store, failed_run_id), **metrics},
                    error=f"{exc.__class__.__name__}: {exc}",
                    repetition=repetition,
                ))

    payload = {
        "suite": suite,
        "runs": max(int(runs), 1),
        "token": token,
        "started_at": started,
        "finished_at": time.time(),
        "output_dir": str(output_path),
        "aggregate": aggregate_results(results),
        "results": [result.to_dict() for result in results],
    }
    _write_text(output_path / f"eval_{suite}_{token}.json", json.dumps(payload, indent=2, ensure_ascii=True))
    return payload


def collect_eval_results(store: StateStore, *, suite: str | None = None) -> list[EvalCaseResult]:
    """Rehydrate primary eval results from a store that supports list_runs."""

    list_runs = getattr(store, "list_runs", None)
    if not callable(list_runs):
        raise ValueError("store does not support list_runs")

    results: list[EvalCaseResult] = []
    for run_id in list_runs():
        capsule = store.load_run(run_id)
        metadata = capsule.manifest.metadata
        if not metadata.get("eval_case"):
            continue
        if metadata.get("eval_case_role", "primary") != "primary":
            continue
        if suite and metadata.get("eval_suite") != suite:
            continue
        metrics = {**capsule_metrics(store, run_id), **dict(metadata.get("eval_metrics", {}))}
        results.append(EvalCaseResult(
            name=str(metadata.get("eval_case", "")),
            run_id=run_id,
            run_ids=[run_id],
            success=bool(metadata.get("eval_success", metrics.get("success", False))),
            duration_ms=float(metadata.get("eval_duration_ms") or metrics.get("duration_ms", 0.0)),
            metrics=metrics,
            repetition=int(metadata.get("eval_repetition", 0) or 0),
        ))
    return sorted(results, key=lambda result: (result.repetition, result.name, result.run_id))


def build_eval_report_payload(store: StateStore, *, suite: str | None = None) -> dict[str, Any]:
    results = collect_eval_results(store, suite=suite)
    suites = sorted({
        str(store.load_run(result.run_id).manifest.metadata.get("eval_suite", ""))
        for result in results
    })
    return {
        "suite": suite or ",".join(suite_name for suite_name in suites if suite_name) or "all",
        "aggregate": aggregate_results(results),
        "results": [result.to_dict() for result in results],
    }


def write_markdown_report(payload: dict[str, Any], path: str | Path) -> Path:
    """Write a concise portfolio-ready markdown eval report."""

    out = Path(path)
    aggregate = payload.get("aggregate", {})
    lines = [
        "# JadeAgent JGX Eval Report",
        "",
        f"- Suite: `{payload.get('suite', 'all')}`",
        f"- Case runs: `{aggregate.get('case_runs', 0)}`",
        f"- Success rate: `{aggregate.get('success_rate', 0.0)}`",
        f"- Task completion rate: `{aggregate.get('task_completion_rate', 0.0)}`",
        f"- Recovery success rate: `{aggregate.get('recovery_success_rate', 0.0)}`",
        f"- Duplicate tool executions: `{aggregate.get('duplicate_tool_executions', 0)}`",
        f"- Secret leak count: `{aggregate.get('secret_leak_count', 0)}`",
        f"- Tokens prompt/completion/total: `{aggregate.get('prompt_tokens', 0)}` / `{aggregate.get('completion_tokens', 0)}` / `{aggregate.get('total_tokens', 0)}`",
        f"- Cost estimate USD: `{aggregate.get('cost_estimate_usd', 0.0)}`",
        f"- Raw baseline p50 ms: raw `{aggregate.get('raw_runtime_ms_p50', 0)}` vs Jade+JGX `{aggregate.get('jade_runtime_ms_p50', 0)}`",
        f"- Runtime p50/p95 ms: `{aggregate.get('runtime_ms_p50', 0)}` / `{aggregate.get('runtime_ms_p95', 0)}`",
        f"- Restore p50/p95 ms: `{aggregate.get('restore_latency_ms_p50', 0)}` / `{aggregate.get('restore_latency_ms_p95', 0)}`",
        "",
        "## Cases",
        "",
        "| Case | Run id | Status | Runtime ms | Events | Snapshots | Key metric |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("results", []):
        metrics = row.get("metrics", {})
        key_metric = ""
        if row["name"] == "tool_idempotency":
            key_metric = f"side_effect_calls={metrics.get('side_effect_calls', '')}; reused={metrics.get('tool_results_reused', '')}"
        elif row["name"] == "crash_recovery":
            key_metric = f"recovered={metrics.get('recovery_success', '')}; attempts={len(metrics.get('attempts', []))}"
        elif row["name"] == "state_restore":
            key_metric = f"restore_ms={metrics.get('restore_latency_ms', '')}"
        elif row["name"] == "compatibility_guard":
            key_metric = f"blocked={metrics.get('blocked_restore_count', '')}"
        elif row["name"] == "raw_call_baseline":
            key_metric = (
                f"raw_ms={metrics.get('raw_runtime_ms', '')}; "
                f"jade_ms={metrics.get('jade_runtime_ms', '')}; "
                f"overhead_ms={metrics.get('state_overhead_ms', '')}"
            )
        elif row["name"] == "mesh_project":
            key_metric = f"files={metrics.get('generated_files', '')}; tests={metrics.get('unit_tests_passed', '')}"
        lines.append(
            f"| `{row['name']}` | `{row['run_id']}` | `{row['success']}` | "
            f"{row['duration_ms']} | {metrics.get('event_count', 0)} | "
            f"{metrics.get('snapshot_count', 0)} | {key_metric} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "These metrics are intentionally about reliability and governance: recovery,",
        "idempotent side effects, restore compatibility, audit completeness, and",
        "state overhead. They are designed for portfolio proof, not synthetic model",
        "leaderboard scoring.",
        "",
    ])
    return _write_text(out, "\n".join(lines))
