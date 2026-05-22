"""Crash recovery demo using SqliteStateStore and mesh retries.

The first worker writes a checkpoint and raises an exception to simulate a
crash. The task store requeues the task. A second worker reads the JGX capsule
from SQLite, resumes from the saved checkpoint, and completes the task.
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jadeagent import JadeStateEvent, JadeStateManifest, SqliteStateStore
from jadeagent.mesh import InMemoryMeshBus, InMemoryTaskStore, MeshNode, MeshRouter, MeshTask, TaskState
from jadeagent.state.snapshot import AgentRuntimeSnapshot


GENERATED_ROOT = ROOT / "examples" / "generated"
REPORT_PATH = ROOT / "docs" / "jgx-crash-recovery-results.md"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def _find_stage(store: SqliteStateStore, run_id: str, stage: str) -> AgentRuntimeSnapshot | None:
    try:
        capsule = store.load_run(run_id)
    except KeyError:
        return None
    for snapshot in reversed(capsule.snapshots):
        if snapshot.metadata.get("stage") == stage:
            return snapshot
    return None


def run_demo() -> dict[str, Any]:
    workflow_id = time.strftime("%Y%m%d_%H%M%S")
    root = GENERATED_ROOT / f"jgx_crash_recovery_{workflow_id}"
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.sqlite3"
    output_path = root / "recovered_artifact.txt"
    run_id = f"crash_recovery_{workflow_id}"

    store = SqliteStateStore(state_path)
    task_store = InMemoryTaskStore()
    router = MeshRouter()
    bus = InMemoryMeshBus()
    attempts: list[dict[str, Any]] = []

    store.create_run(JadeStateManifest(
        run_id=run_id,
        state_kind="mesh_task",
        capability="recover_document",
        agent_id="crash-recovery-demo",
    ))
    store.append_event(run_id, JadeStateEvent(
        event_type="demo_started",
        run_id=run_id,
        phase="NEW",
        message="crash recovery demo started",
    ))

    def handler(task: MeshTask) -> str:
        worker_name = task.metadata.get("worker_name", "unknown")
        draft = _find_stage(store, run_id, "draft_written")
        if draft is None:
            output_path.write_text("phase 1: draft written before crash\n", encoding="utf-8")
            snapshot = AgentRuntimeSnapshot(
                phase="CHECKPOINTING",
                step=1,
                metadata={
                    "stage": "draft_written",
                    "worker": worker_name,
                    "output_path": str(output_path),
                },
            )
            store.save_snapshot(run_id, snapshot)
            store.append_event(run_id, JadeStateEvent(
                event_type="simulated_crash",
                run_id=run_id,
                phase="FAILED",
                step=1,
                actor=worker_name,
                message="worker crashed after writing draft checkpoint",
                payload={"snapshot_id": snapshot.snapshot_id},
            ))
            attempts.append({"worker": worker_name, "action": "crashed_after_checkpoint"})
            raise RuntimeError("simulated crash after checkpoint")

        with output_path.open("a", encoding="utf-8") as handle:
            handle.write("phase 2: recovered worker completed artifact\n")
        snapshot = AgentRuntimeSnapshot(
            phase="COMPLETED",
            step=2,
            metadata={
                "stage": "completed_after_recovery",
                "worker": worker_name,
                "resumed_from_snapshot": draft.snapshot_id,
                "output_path": str(output_path),
            },
        )
        store.save_snapshot(run_id, snapshot)
        store.append_event(run_id, JadeStateEvent(
            event_type="recovered_and_completed",
            run_id=run_id,
            phase="COMPLETED",
            step=2,
            actor=worker_name,
            message="worker resumed from draft checkpoint and completed",
            payload={"snapshot_id": snapshot.snapshot_id},
        ))
        attempts.append({
            "worker": worker_name,
            "action": "resumed_and_completed",
            "resumed_from": draft.snapshot_id,
        })
        return json.dumps({
            "status": "completed",
            "run_id": run_id,
            "output_path": str(output_path),
            "resumed_from": draft.snapshot_id,
        }, ensure_ascii=True)

    coordinator = MeshNode(
        node_id="crash_demo_coordinator",
        capabilities={"coordinate"},
        router=router,
        bus=bus,
        task_store=task_store,
        state_store=store,
        task_handler=lambda task: "coordinator",
    )
    worker_a = MeshNode(
        node_id="worker_before_crash",
        capabilities={"recover_document"},
        router=router,
        bus=bus,
        task_store=task_store,
        state_store=store,
        task_handler=lambda task: handler(_with_worker_name(task, "worker_before_crash")),
    )
    worker_b = MeshNode(
        node_id="worker_after_recovery",
        capabilities={"recover_document"},
        router=router,
        bus=bus,
        task_store=task_store,
        state_store=store,
        task_handler=lambda task: handler(_with_worker_name(task, "worker_after_recovery")),
    )

    task = MeshTask(
        capability="recover_document",
        prompt="Write a two-phase recovered artifact",
        requester=coordinator.node_id,
        metadata={"jgx_run_id": run_id},
        max_attempts=2,
        lease_seconds=2.0,
    )
    task_id = coordinator.submit_task(task)

    worker_a.step()
    first_record = task_store.get(task_id)
    worker_b.step()
    result = coordinator.get_result(task_id)
    final_record = task_store.get(task_id)

    capsule = store.load_run(run_id)
    info = store.inspect(run_id)
    history = [event.to_dict() for event in store.list_events(run_id, limit=50)]
    report = {
        "workflow_id": workflow_id,
        "run_id": run_id,
        "state_path": str(state_path),
        "output_path": str(output_path),
        "task_id": task_id,
        "first_record_state": first_record.state.value if first_record else "",
        "final_record_state": final_record.state.value if final_record else "",
        "result_state": result.state.value if result else "",
        "result_output": result.output if result else "",
        "attempts": attempts,
        "inspect": info,
        "event_types": [event["event_type"] for event in history],
        "snapshot_phases": [snapshot.phase for snapshot in capsule.snapshots],
        "artifact": output_path.read_text(encoding="utf-8") if output_path.exists() else "",
    }
    _write(root / "workflow_result.json", json.dumps(report, indent=2, ensure_ascii=True))
    _write_report(report)
    store.close()
    return report


def _with_worker_name(task: MeshTask, worker_name: str) -> MeshTask:
    task.metadata = dict(task.metadata)
    task.metadata["worker_name"] = worker_name
    return task


def _write_report(report: dict[str, Any]) -> None:
    content = "\n".join([
        "# JGX Crash Recovery Results",
        "",
        "Date: 2026-05-07",
        "",
        "This demo simulates a worker crash after a checkpoint and resumes the",
        "same mesh task from a SQLite-backed JGX state store.",
        "",
        "## Run",
        "",
        f"- Workflow id: `{report['workflow_id']}`",
        f"- Run id: `{report['run_id']}`",
        f"- SQLite state: `{report['state_path']}`",
        f"- Output artifact: `{report['output_path']}`",
        f"- Final task state: `{report['final_record_state']}`",
        f"- Result state: `{report['result_state']}`",
        "",
        "## Attempts",
        "",
        "\n".join(
            f"- `{item['worker']}` -> `{item['action']}`"
            for item in report["attempts"]
        ),
        "",
        "## JGX Inspect",
        "",
        f"- Latest phase: `{report['inspect']['latest_phase']}`",
        f"- Snapshots: `{report['inspect']['snapshot_count']}`",
        f"- Events: `{report['inspect']['event_count']}`",
        "",
        "## Event Types",
        "",
        "```text",
        "\n".join(report["event_types"]),
        "```",
        "",
        "## Artifact",
        "",
        "```text",
        report["artifact"].rstrip(),
        "```",
        "",
    ])
    _write(REPORT_PATH, content)


if __name__ == "__main__":
    result = run_demo()
    print("CRASH_RECOVERY=" + json.dumps({
        "run_id": result["run_id"],
        "state_path": result["state_path"],
        "output_path": result["output_path"],
        "final_record_state": result["final_record_state"],
        "latest_phase": result["inspect"]["latest_phase"],
        "snapshots": result["inspect"]["snapshot_count"],
        "events": result["inspect"]["event_count"],
        "report": str(REPORT_PATH),
    }, ensure_ascii=True, sort_keys=True))
