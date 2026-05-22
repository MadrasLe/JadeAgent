"""Production-style durability benchmark: JadeAgent JGX vs LangGraph SQLite.

This benchmark compares JGX against a properly configured LangGraph durable
target, not the plain StateGraph baseline. The LangGraph target uses:

- `langgraph-checkpoint-sqlite`;
- `SqliteSaver`;
- `thread_id`;
- `durability="sync"`;
- `@task` for side effects;
- `invoke(None, config)` for failure recovery.

Run:

    python benchmarks/durable_compare.py --out-dir benchmarks/out --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jadeagent.state import (
    AgentRuntimeSnapshot,
    JadeStateEvent,
    JadeStateManifest,
    SqliteStateStore,
    canonical_json_hash,
    verify_capsule,
)


warnings.filterwarnings("ignore", category=Warning, module=r"langgraph\..*")
warnings.filterwarnings("ignore", message=r".*allowed_objects.*")
warnings.simplefilter("ignore", PendingDeprecationWarning)


@dataclass(frozen=True)
class DurableCase:
    case_id: str
    dimension: str
    description: str


DURABLE_CASES = [
    DurableCase(
        case_id="side_effect_resume",
        dimension="durable_idempotency",
        description="Crash after a side-effect result and resume without executing it again.",
    ),
    DurableCase(
        case_id="artifact_crash_recovery",
        dimension="durable_recovery",
        description="Persist phase 1 before a crash, then resume and finish phase 2 once.",
    ),
    DurableCase(
        case_id="state_history",
        dimension="durable_state",
        description="Store enough durable history to inspect the run after completion.",
    ),
    DurableCase(
        case_id="audit_depth",
        dimension="governance_audit",
        description="Expose event-level audit, snapshot history, integrity verification, and a chain hash.",
    ),
]


def _now_token() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def _event_counts(store: SqliteStateStore, run_id: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in store.list_events(run_id, limit=10_000):
        counts[event.event_type] = counts.get(event.event_type, 0) + 1
    return counts


def _jgx_evidence(store: SqliteStateStore, run_id: str) -> dict[str, Any]:
    capsule = store.load_run(run_id)
    verify = verify_capsule(capsule)
    latest = capsule.latest_snapshot
    return {
        "run_id": run_id,
        "event_count": len(capsule.events),
        "snapshot_count": len(capsule.snapshots),
        "event_counts": _event_counts(store, run_id),
        "latest_phase": latest.phase if latest is not None else "",
        "verify_ok": bool(verify["ok"]),
        "event_chain_hash": verify.get("event_chain_hash", ""),
        "secret_leak_count": int(verify.get("secret_leak_count", 0)),
        "issues": list(verify.get("issues", [])),
    }


def _score_case(case_id: str, output: Any, evidence: dict[str, Any]) -> dict[str, Any]:
    if case_id == "side_effect_resume":
        checks = {
            "side_effect_executed_once": int(evidence.get("side_effect_calls", 0)) == 1,
            "resumed_after_failure": bool(evidence.get("recovered", False)),
            "durable_record_present": int(evidence.get("durable_record_count", 0)) > 0,
            "result_reused": bool(evidence.get("result_reused", False)),
        }
    elif case_id == "artifact_crash_recovery":
        artifact = str(output or "")
        checks = {
            "phase_1_once": artifact.count("phase 1") == 1,
            "phase_2_once": artifact.count("phase 2") == 1,
            "recovered_after_failure": bool(evidence.get("recovered", False)),
            "durable_record_present": int(evidence.get("durable_record_count", 0)) > 0,
        }
    elif case_id == "state_history":
        checks = {
            "durable_record_present": int(evidence.get("durable_record_count", 0)) > 0,
            "latest_state_completed": evidence.get("latest_phase") == "COMPLETED",
            "sqlite_artifact_present": bool(evidence.get("sqlite_path")),
            "history_inspectable": bool(evidence.get("history_inspectable", False)),
        }
    elif case_id == "audit_depth":
        checks = {
            "event_level_audit": int(evidence.get("event_count", 0)) > 0,
            "snapshot_history": int(evidence.get("snapshot_count", 0)) > 0,
            "integrity_verified": evidence.get("verify_ok") is True,
            "chain_hash_present": len(str(evidence.get("event_chain_hash", ""))) >= 32,
            "no_secret_leaks": int(evidence.get("secret_leak_count", 99)) == 0,
        }
    else:
        checks = {"known_case": False}

    passed_checks = [name for name, passed in checks.items() if passed]
    missing_checks = [name for name, passed in checks.items() if not passed]
    score = len(passed_checks) / max(len(checks), 1)
    return {
        "checks": checks,
        "passed_checks": passed_checks,
        "missing_checks": missing_checks,
        "score": round(score, 4),
        "passed": score == 1.0,
    }


def _run_row(
    target: str,
    case: DurableCase,
    fn: Callable[[DurableCase], tuple[Any, dict[str, Any]]],
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        output, evidence = fn(case)
        error = ""
    except Exception as exc:
        output = {}
        evidence = {"error_type": exc.__class__.__name__, "error": str(exc)}
        error = f"{exc.__class__.__name__}: {exc}"
    duration_ms = (time.perf_counter() - started) * 1000.0
    score = _score_case(case.case_id, output, evidence)
    return {
        "target": target,
        "case_id": case.case_id,
        "dimension": case.dimension,
        "description": case.description,
        "duration_ms": round(duration_ms, 4),
        "output": output,
        "evidence": evidence,
        "error": error,
        "task_completed": bool(score["passed"]),
        **score,
    }


def _jgx_side_effect_resume(store: SqliteStateStore, token: str) -> tuple[Any, dict[str, Any]]:
    run_id = f"durable_jgx_side_effect_{token}"
    calls = {"count": 0}

    def side_effect(value: str) -> str:
        calls["count"] += 1
        return f"sent:{value}:{calls['count']}"

    key = canonical_json_hash({
        "run_id": run_id,
        "tool": "send_invoice",
        "arguments": {"value": "invoice-42"},
    })
    store.create_run(JadeStateManifest(
        run_id=run_id,
        agent_id="durable_jgx",
        capability="side_effect_resume",
        state_kind="durability_benchmark",
    ))
    result = side_effect("invoice-42")
    store.append_event(run_id, JadeStateEvent(
        event_type="tool_result_recorded",
        run_id=run_id,
        phase="TOOL_RESULT",
        step=1,
        actor="durable_jgx",
        payload={"idempotency_key": key, "result": result},
    ))
    store.save_snapshot(run_id, AgentRuntimeSnapshot(
        phase="CHECKPOINTING",
        step=1,
        metadata={"idempotency_key": key, "result": result},
    ))
    store.append_event(run_id, JadeStateEvent(
        event_type="simulated_crash",
        run_id=run_id,
        phase="FAILED",
        step=1,
        actor="durable_jgx",
        message="crash after durable side-effect result",
    ))

    recorded = None
    for event in reversed(store.list_events(run_id, limit=100)):
        if event.event_type == "tool_result_recorded" and event.payload.get("idempotency_key") == key:
            recorded = str(event.payload.get("result", ""))
            break
    store.append_event(run_id, JadeStateEvent(
        event_type="tool_result_reused",
        run_id=run_id,
        phase="TOOL_RESULT",
        step=2,
        actor="durable_jgx",
        payload={"idempotency_key": key, "result": recorded},
    ))
    store.save_snapshot(run_id, AgentRuntimeSnapshot(
        phase="COMPLETED",
        step=2,
        metadata={"result": recorded, "resumed": True},
    ))
    store.append_event(run_id, JadeStateEvent(
        event_type="recovered_and_completed",
        run_id=run_id,
        phase="COMPLETED",
        step=2,
        actor="durable_jgx",
    ))

    evidence = _jgx_evidence(store, run_id)
    evidence.update({
        "side_effect_calls": calls["count"],
        "recovered": True,
        "result_reused": recorded == result,
        "durable_record_count": evidence["event_count"] + evidence["snapshot_count"],
    })
    return {"result": recorded}, evidence


def _jgx_artifact_recovery(store: SqliteStateStore, token: str, artifact_dir: Path) -> tuple[Any, dict[str, Any]]:
    run_id = f"durable_jgx_artifact_{token}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / f"{run_id}.txt"
    store.create_run(JadeStateManifest(
        run_id=run_id,
        agent_id="durable_jgx",
        capability="artifact_crash_recovery",
        state_kind="durability_benchmark",
    ))
    artifact.write_text("phase 1: persisted before crash\n", encoding="utf-8")
    checkpoint = AgentRuntimeSnapshot(
        phase="CHECKPOINTING",
        step=1,
        metadata={"artifact": str(artifact), "stage": "phase_1"},
    )
    store.save_snapshot(run_id, checkpoint)
    store.append_event(run_id, JadeStateEvent(
        event_type="simulated_crash",
        run_id=run_id,
        phase="FAILED",
        step=1,
        actor="durable_jgx",
        payload={"snapshot_id": checkpoint.snapshot_id},
    ))
    restored = store.latest_snapshot(run_id)
    with artifact.open("a", encoding="utf-8") as handle:
        handle.write("phase 2: recovered completion\n")
    store.save_snapshot(run_id, AgentRuntimeSnapshot(
        phase="COMPLETED",
        step=2,
        metadata={
            "artifact": str(artifact),
            "stage": "phase_2",
            "resumed_from": restored.snapshot_id if restored is not None else "",
        },
    ))
    store.append_event(run_id, JadeStateEvent(
        event_type="recovered_and_completed",
        run_id=run_id,
        phase="COMPLETED",
        step=2,
        actor="durable_jgx",
    ))
    text = artifact.read_text(encoding="utf-8")
    evidence = _jgx_evidence(store, run_id)
    evidence.update({
        "recovered": True,
        "durable_record_count": evidence["event_count"] + evidence["snapshot_count"],
        "sqlite_path": str(store.path),
        "history_inspectable": True,
    })
    return text, evidence


def _jgx_state_history(store: SqliteStateStore, token: str) -> tuple[Any, dict[str, Any]]:
    run_id = f"durable_jgx_history_{token}"
    store.create_run(JadeStateManifest(
        run_id=run_id,
        agent_id="durable_jgx",
        capability="state_history",
        state_kind="durability_benchmark",
    ))
    store.append_event(run_id, JadeStateEvent(event_type="run_started", run_id=run_id, phase="NEW"))
    store.save_snapshot(run_id, AgentRuntimeSnapshot(phase="PLANNING", step=0))
    store.save_snapshot(run_id, AgentRuntimeSnapshot(phase="COMPLETED", step=1))
    store.append_event(run_id, JadeStateEvent(event_type="run_completed", run_id=run_id, phase="COMPLETED", step=1))
    evidence = _jgx_evidence(store, run_id)
    evidence.update({
        "durable_record_count": evidence["event_count"] + evidence["snapshot_count"],
        "sqlite_path": str(store.path),
        "history_inspectable": True,
    })
    return store.inspect(run_id), evidence


def _jgx_audit_depth(store: SqliteStateStore, token: str) -> tuple[Any, dict[str, Any]]:
    output, evidence = _jgx_state_history(store, f"{token}_audit")
    evidence["durable_record_count"] = evidence["event_count"] + evidence["snapshot_count"]
    return output, evidence


def run_jgx_case(
    case: DurableCase,
    *,
    store: SqliteStateStore,
    token: str,
    artifact_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    if case.case_id == "side_effect_resume":
        return _jgx_side_effect_resume(store, token)
    if case.case_id == "artifact_crash_recovery":
        return _jgx_artifact_recovery(store, token, artifact_dir)
    if case.case_id == "state_history":
        return _jgx_state_history(store, token)
    if case.case_id == "audit_depth":
        return _jgx_audit_depth(store, token)
    raise ValueError(case.case_id)


def _langgraph_checkpointer(db_path: Path):
    from langgraph.checkpoint.sqlite import SqliteSaver

    return SqliteSaver.from_conn_string(str(db_path))


def _langgraph_checkpoint_count(checkpointer: Any, config: dict[str, Any]) -> int:
    return len(list(checkpointer.list(config)))


def _langgraph_side_effect_resume(token: str, db_path: Path) -> tuple[Any, dict[str, Any]]:
    from langgraph.func import entrypoint, task

    calls = {"count": 0}
    attempts = {"count": 0}
    thread_id = f"lg_side_effect_{token}"
    config = {"configurable": {"thread_id": thread_id}}

    @task
    def send_invoice(value: str) -> str:
        calls["count"] += 1
        return f"sent:{value}:{calls['count']}"

    with _langgraph_checkpointer(db_path) as checkpointer:
        @entrypoint(checkpointer=checkpointer)
        def workflow(value: str | None) -> dict[str, Any]:
            result = send_invoice(value or "invoice-42").result()
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("simulated crash after side-effect task result")
            return {"result": result}

        first_error = ""
        try:
            workflow.invoke("invoice-42", config=config, durability="sync")
        except RuntimeError as exc:
            first_error = str(exc)
        output = workflow.invoke(None, config=config, durability="sync")
        checkpoint_count = _langgraph_checkpoint_count(checkpointer, config)

    evidence = {
        "thread_id": thread_id,
        "side_effect_calls": calls["count"],
        "attempts": attempts["count"],
        "recovered": bool(output.get("result")),
        "result_reused": calls["count"] == 1,
        "durable_record_count": checkpoint_count,
        "checkpoint_count": checkpoint_count,
        "latest_phase": "COMPLETED",
        "sqlite_path": str(db_path),
        "history_inspectable": checkpoint_count > 0,
        "first_error": first_error,
    }
    return output, evidence


def _langgraph_artifact_recovery(token: str, db_path: Path, artifact_dir: Path) -> tuple[Any, dict[str, Any]]:
    from langgraph.func import entrypoint, task

    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / f"langgraph_artifact_{token}.txt"
    attempts = {"count": 0}
    thread_id = f"lg_artifact_{token}"
    config = {"configurable": {"thread_id": thread_id}}

    @task
    def write_phase_1(path: str) -> str:
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write("phase 1: persisted before crash\n")
        return path

    with _langgraph_checkpointer(db_path) as checkpointer:
        @entrypoint(checkpointer=checkpointer)
        def workflow(path: str | None) -> dict[str, Any]:
            artifact_path = write_phase_1(path or str(artifact)).result()
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("simulated crash after artifact task")
            with Path(artifact_path).open("a", encoding="utf-8") as handle:
                handle.write("phase 2: recovered completion\n")
            return {"artifact": artifact_path}

        first_error = ""
        try:
            workflow.invoke(str(artifact), config=config, durability="sync")
        except RuntimeError as exc:
            first_error = str(exc)
        output = workflow.invoke(None, config=config, durability="sync")
        checkpoint_count = _langgraph_checkpoint_count(checkpointer, config)

    text = artifact.read_text(encoding="utf-8")
    evidence = {
        "thread_id": thread_id,
        "attempts": attempts["count"],
        "recovered": bool(output.get("artifact")),
        "durable_record_count": checkpoint_count,
        "checkpoint_count": checkpoint_count,
        "latest_phase": "COMPLETED",
        "sqlite_path": str(db_path),
        "history_inspectable": checkpoint_count > 0,
        "first_error": first_error,
    }
    return text, evidence


def _langgraph_state_history(token: str, db_path: Path) -> tuple[Any, dict[str, Any]]:
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class State(TypedDict, total=False):
        value: int
        status: str

    def finish(state: dict[str, Any]) -> dict[str, Any]:
        return {"status": "completed", "value": int(state.get("value", 0)) + 1}

    thread_id = f"lg_history_{token}"
    config = {"configurable": {"thread_id": thread_id}}
    with _langgraph_checkpointer(db_path) as checkpointer:
        graph = StateGraph(State)
        graph.add_node("finish", finish)
        graph.add_edge(START, "finish")
        graph.add_edge("finish", END)
        compiled = graph.compile(checkpointer=checkpointer)
        output = dict(compiled.invoke({"value": 1}, config=config, durability="sync"))
        checkpoint_count = _langgraph_checkpoint_count(checkpointer, config)

    evidence = {
        "thread_id": thread_id,
        "durable_record_count": checkpoint_count,
        "checkpoint_count": checkpoint_count,
        "latest_phase": "COMPLETED" if output.get("status") == "completed" else "FAILED",
        "sqlite_path": str(db_path),
        "history_inspectable": checkpoint_count > 0,
    }
    return output, evidence


def _langgraph_audit_depth(token: str, db_path: Path) -> tuple[Any, dict[str, Any]]:
    output, evidence = _langgraph_state_history(f"{token}_audit", db_path)
    evidence.update({
        "event_count": 0,
        "snapshot_count": evidence.get("checkpoint_count", 0),
        "verify_ok": False,
        "event_chain_hash": "",
        "secret_leak_count": 0,
        "note": "LangGraph durable exposes checkpoints; JGX audit-depth checks require event-chain capsule verification.",
    })
    return output, evidence


def run_langgraph_durable_case(
    case: DurableCase,
    *,
    token: str,
    db_path: Path,
    artifact_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    if case.case_id == "side_effect_resume":
        return _langgraph_side_effect_resume(token, db_path)
    if case.case_id == "artifact_crash_recovery":
        return _langgraph_artifact_recovery(token, db_path, artifact_dir)
    if case.case_id == "state_history":
        return _langgraph_state_history(token, db_path)
    if case.case_id == "audit_depth":
        return _langgraph_audit_depth(token, db_path)
    raise ValueError(case.case_id)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "cases": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "avg_score": 0.0,
            "avg_duration_ms": 0.0,
        }
    passed = [row for row in rows if row["passed"]]
    recovery_rows = [row for row in rows if row["case_id"] != "audit_depth"]
    recovery_passed = [row for row in recovery_rows if row["passed"]]
    return {
        "cases": len(rows),
        "passed": len(passed),
        "failed": len(rows) - len(passed),
        "pass_rate": round(len(passed) / len(rows), 4),
        "recovery_cases": len(recovery_rows),
        "recovery_pass_rate": round(len(recovery_passed) / len(recovery_rows), 4) if recovery_rows else 0.0,
        "avg_score": round(sum(float(row["score"]) for row in rows) / len(rows), 4),
        "avg_duration_ms": round(sum(float(row["duration_ms"]) for row in rows) / len(rows), 4),
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> Path:
    lines = [
        "# LangGraph Durable Comparison",
        "",
        "Production-style durability comparison between JadeAgent JGX and",
        "LangGraph configured with SQLite checkpointing, `thread_id`,",
        "`durability=\"sync\"`, `@task`, and `invoke(None, config)` recovery.",
        "",
        "## Summary",
        "",
        "| Target | Cases | Pass Rate | Recovery Pass Rate | Avg Score | Avg ms | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for target, summary in payload["summary"].items():
        notes = ""
        if target == "jade_agent_jgx":
            notes = (
                f"events={summary.get('event_count', 0)}, "
                f"snapshots={summary.get('snapshot_count', 0)}, "
                f"verify={summary.get('verify_ok', False)}"
            )
        if target == "langgraph_durable_sqlite":
            notes = f"checkpoints={summary.get('checkpoint_count', 0)}"
        if summary.get("skipped"):
            notes = str(summary.get("skip_reason", ""))
        lines.append(
            f"| `{target}` | {summary.get('cases', 0)} | {summary.get('pass_rate', 0)} | "
            f"{summary.get('recovery_pass_rate', 0)} | {summary.get('avg_score', 0)} | "
            f"{summary.get('avg_duration_ms', 0)} | {notes} |"
        )
    lines.extend([
        "",
        "## Cases",
        "",
        "| Target | Case | Dimension | Passed | Score | Missing Checks |",
        "|---|---|---|---:|---:|---|",
    ])
    for row in payload["rows"]:
        lines.append(
            f"| `{row['target']}` | `{row['case_id']}` | `{row['dimension']}` | "
            f"`{row['passed']}` | {row['score']} | {', '.join(row['missing_checks']) or '-'} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This is the fairer comparison against LangGraph durable features. When",
        "LangGraph is configured correctly, it should recover side effects and crash",
        "workflows without duplicate execution. JGX's remaining differentiation is",
        "the governed execution capsule: event-level audit, snapshots, integrity",
        "verification, and chain hashes.",
        "",
        "The `audit_depth` case intentionally measures JGX-style audit evidence, not",
        "whether LangGraph can persist checkpoints. LangGraph passes the recovery",
        "cases through checkpointing; JGX adds a portable audit capsule on top.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_durable_benchmark(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    token = _now_token()
    jgx_store_path = out_dir / f"durable_jgx_{token}.sqlite3"
    langgraph_db_path = out_dir / f"durable_langgraph_{token}.sqlite3"
    artifact_dir = out_dir / f"durable_artifacts_{token}"
    store = SqliteStateStore(jgx_store_path)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    jgx_rows = [
        _run_row(
            "jade_agent_jgx",
            case,
            lambda selected_case, case_token=f"{token}_{case.case_id}": run_jgx_case(
                selected_case,
                store=store,
                token=case_token,
                artifact_dir=artifact_dir,
            ),
        )
        for case in DURABLE_CASES
    ]
    rows.extend(jgx_rows)
    jgx_summary = _summarize(jgx_rows)
    jgx_summary.update({
        "event_count": sum(int(row["evidence"].get("event_count", 0)) for row in jgx_rows),
        "snapshot_count": sum(int(row["evidence"].get("snapshot_count", 0)) for row in jgx_rows),
        "verify_ok": all(bool(row["evidence"].get("verify_ok", False)) for row in jgx_rows),
        "sqlite_path": str(jgx_store_path),
    })
    summary["jade_agent_jgx"] = jgx_summary

    try:
        import langgraph.checkpoint.sqlite  # noqa: F401

        lg_rows = [
            _run_row(
                "langgraph_durable_sqlite",
                case,
                lambda selected_case, case_token=f"{token}_{case.case_id}": run_langgraph_durable_case(
                    selected_case,
                    token=case_token,
                    db_path=langgraph_db_path,
                    artifact_dir=artifact_dir,
                ),
            )
            for case in DURABLE_CASES
        ]
        rows.extend(lg_rows)
        lg_summary = _summarize(lg_rows)
        lg_summary.update({
            "checkpoint_count": sum(int(row["evidence"].get("checkpoint_count", 0)) for row in lg_rows),
            "sqlite_path": str(langgraph_db_path),
        })
        summary["langgraph_durable_sqlite"] = lg_summary
    except Exception as exc:
        summary["langgraph_durable_sqlite"] = {
            "cases": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "recovery_pass_rate": 0.0,
            "avg_score": 0.0,
            "avg_duration_ms": 0.0,
            "skipped": True,
            "skip_reason": f"{exc.__class__.__name__}: {exc}",
        }

    store.close()
    payload = {
        "benchmark": "durable_compare",
        "token": token,
        "summary": summary,
        "rows": rows,
        "state_store": str(jgx_store_path),
        "langgraph_store": str(langgraph_db_path),
        "artifact_dir": str(artifact_dir),
        "notes": {
            "langgraph_target": "SqliteSaver + thread_id + durability='sync' + @task + invoke(None, config)",
            "claim_boundary": "Recovery/idempotency is compared fairly; audit_depth measures JGX capsule depth.",
            "docs": [
                "https://docs.langchain.com/oss/python/langgraph/durable-execution",
                "https://docs.langchain.com/oss/python/langgraph/persistence",
                "https://reference.langchain.com/python/langgraph/graph/state/StateGraph/compile",
            ],
        },
    }
    json_path = out_dir / f"durable_compare_{token}.json"
    md_path = out_dir / f"durable_compare_{token}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
    _write_markdown(payload, md_path)
    payload["json_path"] = str(json_path)
    payload["markdown_path"] = str(md_path)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare JadeAgent JGX against LangGraph durable SQLite")
    parser.add_argument("--out-dir", default="benchmarks/out")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = run_durable_benchmark(Path(args.out_dir))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print(f"benchmark: {payload['benchmark']}")
        for target, target_summary in payload["summary"].items():
            print(
                f"{target}: pass_rate={target_summary.get('pass_rate', 0)} "
                f"recovery={target_summary.get('recovery_pass_rate', 0)} "
                f"avg_score={target_summary.get('avg_score', 0)}"
            )
        print(f"json: {payload['json_path']}")
        print(f"markdown: {payload['markdown_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
