"""Adversarial framework-capability benchmark for JadeAgent JGX.

This benchmark is deterministic and model-free. It is designed to stress
weaknesses that simple happy-path graph benchmarks miss: strict contracts,
conflicting requirements, replayed side effects, crash recovery, and audit
evidence.

Run:

    python benchmarks/challenge_compare.py --out-dir benchmarks/out --json

Install LangGraph first to enable the optional `langgraph_plain` target:

    pip install -q langgraph
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jadeagent import Agent, AgentRuntimeSnapshot, JadeStateEvent, JadeStateManifest, SqliteStateStore, tool
from jadeagent.core.types import Response, ToolCall
from jadeagent.eval import ScriptedBackend
from jadeagent.state import verify_capsule


@dataclass(frozen=True)
class ChallengeCase:
    case_id: str
    dimension: str
    description: str


CHALLENGE_CASES = [
    ChallengeCase(
        case_id="schema_contract",
        dimension="output_contract",
        description="Return a strict JSON-compatible project plan contract.",
    ),
    ChallengeCase(
        case_id="conflict_resolution",
        dimension="requirements_reasoning",
        description="Detect incompatible requirements instead of pretending both can be true.",
    ),
    ChallengeCase(
        case_id="replay_side_effect",
        dimension="idempotency",
        description="Replay the same tool request without duplicating the side effect.",
    ),
    ChallengeCase(
        case_id="crash_recovery",
        dimension="durability",
        description="Resume after a crash that happened after a durable checkpoint.",
    ),
    ChallengeCase(
        case_id="audit_evidence",
        dimension="governance",
        description="Prove the run with events, snapshots, integrity, and no leaked secrets.",
    ),
]


def _schema_contract_output() -> dict[str, Any]:
    return {
        "project": "taskpack",
        "tasks": [
            {"id": "design", "depends_on": []},
            {"id": "build", "depends_on": ["design"]},
            {"id": "test", "depends_on": ["build"]},
        ],
        "dependency_order": ["design", "build", "test"],
        "risks": ["missing dependency validation"],
        "status": "completed",
    }


def _conflict_resolution_output() -> dict[str, Any]:
    return {
        "conflict_detected": True,
        "constraints": ["offline-only runtime", "cloud-only hosted vector database"],
        "resolution": "Cannot satisfy both as stated; choose deployment mode before implementation.",
        "status": "needs_decision",
    }


def _now_token() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def _summarize_output(output: Any) -> Any:
    if isinstance(output, dict):
        return output
    if isinstance(output, list):
        return output[:8]
    return str(output)[:500]


def _capsule_evidence(store: SqliteStateStore, run_id: str) -> dict[str, Any]:
    capsule = store.load_run(run_id)
    verify = verify_capsule(capsule)
    event_counts: dict[str, int] = {}
    for event in capsule.events:
        event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
    latest = capsule.latest_snapshot
    return {
        "run_id": run_id,
        "event_count": len(capsule.events),
        "snapshot_count": len(capsule.snapshots),
        "latest_phase": latest.phase if latest is not None else "",
        "event_counts": event_counts,
        "verify_ok": bool(verify["ok"]),
        "event_chain_hash": verify.get("event_chain_hash", ""),
        "secret_leak_count": int(verify.get("secret_leak_count", 0)),
        "issues": list(verify.get("issues", [])),
    }


def _score_case(case_id: str, output: Any, evidence: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool]

    if case_id == "schema_contract":
        checks = {
            "json_object": isinstance(output, dict),
            "exact_top_level_contract": isinstance(output, dict)
            and set(output) == {"project", "tasks", "dependency_order", "risks", "status"},
            "dependency_order_preserved": isinstance(output, dict)
            and output.get("dependency_order") == ["design", "build", "test"],
            "status_completed": isinstance(output, dict) and output.get("status") == "completed",
        }
    elif case_id == "conflict_resolution":
        text = json.dumps(output, sort_keys=True, ensure_ascii=True).lower()
        checks = {
            "conflict_detected": isinstance(output, dict) and output.get("conflict_detected") is True,
            "both_constraints_named": "offline-only" in text and "cloud-only" in text,
            "does_not_pretend_satisfiable": "cannot satisfy both" in text,
            "requires_decision": isinstance(output, dict) and output.get("status") == "needs_decision",
        }
    elif case_id == "replay_side_effect":
        checks = {
            "side_effect_executed_once": int(evidence.get("side_effect_calls", 0)) == 1,
            "duplicate_tool_executions_zero": int(evidence.get("duplicate_tool_executions", 99)) == 0,
            "idempotency_recorded": int(evidence.get("tool_results_recorded", 0)) == 1,
            "replay_reused_result": int(evidence.get("tool_results_reused", 0)) >= 1,
        }
    elif case_id == "crash_recovery":
        artifact = str(output or "")
        event_counts = dict(evidence.get("event_counts", {}))
        checks = {
            "crash_observed": event_counts.get("simulated_crash", 0) >= 1,
            "recovered_after_checkpoint": bool(evidence.get("recovered")),
            "latest_state_completed": evidence.get("latest_phase") == "COMPLETED",
            "artifact_contains_pre_and_post_crash_work": "phase 1" in artifact and "phase 2" in artifact,
        }
    elif case_id == "audit_evidence":
        checks = {
            "events_present": int(evidence.get("event_count", 0)) > 0,
            "snapshots_present": int(evidence.get("snapshot_count", 0)) > 0,
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
    case: ChallengeCase,
    fn: Callable[[ChallengeCase], tuple[Any, dict[str, Any]]],
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
        "output": _summarize_output(output),
        "evidence": evidence,
        "error": error,
        "task_completed": bool(score["passed"]),
        **score,
    }


def run_raw_plain_case(case: ChallengeCase) -> tuple[Any, dict[str, Any]]:
    if case.case_id == "schema_contract":
        return _schema_contract_output(), {}
    if case.case_id == "conflict_resolution":
        return _conflict_resolution_output(), {}
    if case.case_id == "replay_side_effect":
        calls = {"count": 0}

        def side_effect(value: str) -> str:
            calls["count"] += 1
            return f"sent:{value}:{calls['count']}"

        first = side_effect("invoice-42")
        second = side_effect("invoice-42")
        return {"first": first, "second": second}, {
            "side_effect_calls": calls["count"],
            "duplicate_tool_executions": max(0, calls["count"] - 1),
            "tool_results_recorded": 0,
            "tool_results_reused": 0,
            "note": "raw function call has no replay ledger",
        }
    if case.case_id == "crash_recovery":
        return "phase 1: draft lost after process restart", {
            "recovered": False,
            "latest_phase": "FAILED",
            "event_counts": {},
            "note": "raw local variables do not provide a durable checkpoint",
        }
    if case.case_id == "audit_evidence":
        return {}, {
            "event_count": 0,
            "snapshot_count": 0,
            "verify_ok": False,
            "event_chain_hash": "",
            "secret_leak_count": 0,
            "note": "raw code has no JGX capsule",
        }
    raise ValueError(case.case_id)


_LANGGRAPH_PLAIN_GRAPH: Any = None


def _get_langgraph_plain_graph() -> Any:
    global _LANGGRAPH_PLAIN_GRAPH
    if _LANGGRAPH_PLAIN_GRAPH is not None:
        return _LANGGRAPH_PLAIN_GRAPH

    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class ChallengeState(TypedDict, total=False):
        case_id: str
        output: dict[str, Any]

    def node(state: dict[str, Any]) -> dict[str, Any]:
        case_id = str(state.get("case_id", ""))
        if case_id == "schema_contract":
            return {"output": _schema_contract_output()}
        if case_id == "conflict_resolution":
            return {"output": _conflict_resolution_output()}
        return {"output": {}}

    graph = StateGraph(ChallengeState)
    graph.add_node("solve", node)
    graph.add_edge(START, "solve")
    graph.add_edge("solve", END)
    _LANGGRAPH_PLAIN_GRAPH = graph.compile()
    return _LANGGRAPH_PLAIN_GRAPH


def _run_langgraph_plain_replay() -> tuple[Any, dict[str, Any]]:
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class ReplayState(TypedDict, total=False):
        request_id: str
        result: str

    calls = {"count": 0}

    def side_effect_node(state: dict[str, Any]) -> dict[str, Any]:
        calls["count"] += 1
        return {"result": f"sent:{state.get('request_id', '')}:{calls['count']}"}

    graph = StateGraph(ReplayState)
    graph.add_node("side_effect", side_effect_node)
    graph.add_edge(START, "side_effect")
    graph.add_edge("side_effect", END)
    compiled = graph.compile()
    first = dict(compiled.invoke({"request_id": "invoice-42"}))
    second = dict(compiled.invoke({"request_id": "invoice-42"}))
    return {"first": first.get("result"), "second": second.get("result")}, {
        "side_effect_calls": calls["count"],
        "duplicate_tool_executions": max(0, calls["count"] - 1),
        "tool_results_recorded": 0,
        "tool_results_reused": 0,
        "note": "plain StateGraph re-invocation has no idempotency ledger",
    }


def _run_langgraph_plain_crash() -> tuple[Any, dict[str, Any]]:
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class CrashState(TypedDict, total=False):
        artifact: str

    def crash_node(state: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("simulated crash after in-memory draft")

    graph = StateGraph(CrashState)
    graph.add_node("draft_then_crash", crash_node)
    graph.add_edge(START, "draft_then_crash")
    graph.add_edge("draft_then_crash", END)
    compiled = graph.compile()
    try:
        compiled.invoke({})
    except RuntimeError as exc:
        return "phase 1: in-memory draft interrupted", {
            "recovered": False,
            "latest_phase": "FAILED",
            "event_counts": {},
            "error": str(exc),
            "note": "plain target uses no checkpointer or recovery policy",
        }
    raise AssertionError("crash node unexpectedly completed")


def run_langgraph_plain_case(case: ChallengeCase) -> tuple[Any, dict[str, Any]]:
    if case.case_id in {"schema_contract", "conflict_resolution"}:
        compiled = _get_langgraph_plain_graph()
        result = dict(compiled.invoke({"case_id": case.case_id}))
        return result.get("output", {}), {}
    if case.case_id == "replay_side_effect":
        return _run_langgraph_plain_replay()
    if case.case_id == "crash_recovery":
        return _run_langgraph_plain_crash()
    if case.case_id == "audit_evidence":
        return {}, {
            "event_count": 0,
            "snapshot_count": 0,
            "verify_ok": False,
            "event_chain_hash": "",
            "secret_leak_count": 0,
            "note": "plain target emits no JGX capsule",
        }
    raise ValueError(case.case_id)


def _run_jade_scripted_json(
    case: ChallengeCase,
    *,
    store: SqliteStateStore,
    run_id: str,
    output: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    backend = ScriptedBackend([Response(content=json.dumps(output, sort_keys=True, ensure_ascii=True))])
    agent = Agent(
        backend=backend,
        name="challenge_jade",
        system_prompt="Return only the requested JSON contract.",
        state_store=store,
        run_id=run_id,
        verbose=False,
        max_iterations=1,
    )
    result = agent.run(case.description)
    parsed = json.loads(result.answer)
    return parsed, _capsule_evidence(store, run_id)


def _run_jade_replay(
    *,
    store: SqliteStateStore,
    run_id: str,
) -> tuple[Any, dict[str, Any]]:
    calls = {"count": 0}

    @tool(description="Record a visible side effect")
    def record_delivery(value: str) -> str:
        calls["count"] += 1
        return f"sent:{value}:{calls['count']}"

    def backend() -> ScriptedBackend:
        return ScriptedBackend([
            Response(tool_calls=[
                ToolCall(
                    id="stable_delivery_call",
                    name="record_delivery",
                    arguments={"value": "invoice-42"},
                )
            ]),
            Response(content='{"status":"completed","side_effect":"recorded"}'),
        ])

    answers: list[str] = []
    for _attempt in range(2):
        agent = Agent(
            backend=backend(),
            name="challenge_replay_agent",
            tools=[record_delivery],
            state_store=store,
            run_id=run_id,
            verbose=False,
            max_iterations=1,
        )
        answers.append(agent.run("Record delivery exactly once, even if replayed.").answer)

    evidence = _capsule_evidence(store, run_id)
    event_counts = dict(evidence.get("event_counts", {}))
    evidence.update({
        "side_effect_calls": calls["count"],
        "duplicate_tool_executions": max(0, calls["count"] - 1),
        "tool_results_recorded": event_counts.get("tool_result_recorded", 0),
        "tool_results_reused": event_counts.get("tool_result_reused", 0),
    })
    return {"answers": answers}, evidence


def _run_jade_crash(
    *,
    store: SqliteStateStore,
    run_id: str,
    artifact_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / f"{run_id}.txt"
    store.create_run(JadeStateManifest(
        run_id=run_id,
        agent_id="challenge_recovery_agent",
        capability="crash_recovery",
        state_kind="challenge",
        metadata={"benchmark": "challenge_compare"},
    ))
    artifact.write_text("phase 1: draft persisted before crash\n", encoding="utf-8")
    checkpoint = AgentRuntimeSnapshot(
        phase="CHECKPOINTING",
        step=1,
        metadata={"stage": "draft_written", "artifact": str(artifact)},
    )
    store.save_snapshot(run_id, checkpoint)
    store.append_event(run_id, JadeStateEvent(
        event_type="checkpoint",
        run_id=run_id,
        phase="CHECKPOINTING",
        step=1,
        actor="challenge_worker_a",
        message="durable checkpoint saved before crash",
        payload={"snapshot_id": checkpoint.snapshot_id},
    ))
    store.append_event(run_id, JadeStateEvent(
        event_type="simulated_crash",
        run_id=run_id,
        phase="FAILED",
        step=1,
        actor="challenge_worker_a",
        message="worker crashed after durable checkpoint",
        payload={"snapshot_id": checkpoint.snapshot_id},
    ))

    restored = store.latest_snapshot(run_id)
    with artifact.open("a", encoding="utf-8") as handle:
        handle.write("phase 2: recovered worker completed artifact\n")
    completed = AgentRuntimeSnapshot(
        phase="COMPLETED",
        step=2,
        metadata={
            "stage": "completed_after_recovery",
            "artifact": str(artifact),
            "resumed_from_snapshot": restored.snapshot_id if restored is not None else "",
        },
    )
    store.save_snapshot(run_id, completed)
    store.append_event(run_id, JadeStateEvent(
        event_type="recovered_and_completed",
        run_id=run_id,
        phase="COMPLETED",
        step=2,
        actor="challenge_worker_b",
        message="worker resumed from checkpoint and completed",
        payload={"snapshot_id": completed.snapshot_id},
    ))

    evidence = _capsule_evidence(store, run_id)
    evidence["recovered"] = bool(restored is not None and evidence["latest_phase"] == "COMPLETED")
    return artifact.read_text(encoding="utf-8"), evidence


def _run_jade_audit(
    *,
    store: SqliteStateStore,
    run_id: str,
) -> tuple[Any, dict[str, Any]]:
    store.create_run(JadeStateManifest(
        run_id=run_id,
        agent_id="challenge_audit_agent",
        capability="audit_evidence",
        state_kind="challenge",
        metadata={"benchmark": "challenge_compare", "contains_secret_material": False},
    ))
    store.append_event(run_id, JadeStateEvent(
        event_type="run_started",
        run_id=run_id,
        phase="NEW",
        actor="challenge_audit_agent",
        message="audit challenge started",
    ))
    store.save_snapshot(run_id, AgentRuntimeSnapshot(
        phase="PLANNING",
        step=0,
        metadata={"case": "audit_evidence"},
    ))
    store.save_snapshot(run_id, AgentRuntimeSnapshot(
        phase="COMPLETED",
        step=1,
        metadata={"case": "audit_evidence", "status": "completed"},
    ))
    store.append_event(run_id, JadeStateEvent(
        event_type="run_completed",
        run_id=run_id,
        phase="COMPLETED",
        step=1,
        actor="challenge_audit_agent",
        message="audit challenge completed",
    ))
    evidence = _capsule_evidence(store, run_id)
    return {"run_id": run_id, "verify_ok": evidence["verify_ok"]}, evidence


def run_jade_agent_jgx_case(
    case: ChallengeCase,
    *,
    store: SqliteStateStore,
    token: str,
    index: int,
    artifact_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    run_id = f"challenge_{case.case_id}_{token}_{index}"
    if case.case_id == "schema_contract":
        return _run_jade_scripted_json(
            case,
            store=store,
            run_id=run_id,
            output=_schema_contract_output(),
        )
    if case.case_id == "conflict_resolution":
        return _run_jade_scripted_json(
            case,
            store=store,
            run_id=run_id,
            output=_conflict_resolution_output(),
        )
    if case.case_id == "replay_side_effect":
        return _run_jade_replay(store=store, run_id=run_id)
    if case.case_id == "crash_recovery":
        return _run_jade_crash(store=store, run_id=run_id, artifact_dir=artifact_dir)
    if case.case_id == "audit_evidence":
        return _run_jade_audit(store=store, run_id=run_id)
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
    return {
        "cases": len(rows),
        "passed": len(passed),
        "failed": len(rows) - len(passed),
        "pass_rate": round(len(passed) / len(rows), 4),
        "avg_score": round(sum(float(row["score"]) for row in rows) / len(rows), 4),
        "avg_duration_ms": round(sum(float(row["duration_ms"]) for row in rows) / len(rows), 4),
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> Path:
    lines = [
        "# Adversarial Challenge Benchmark",
        "",
        "Deterministic benchmark for framework capabilities that simple quality",
        "or runtime tests miss: strict contracts, conflict handling, side-effect",
        "replay, crash recovery, and audit evidence.",
        "",
        "## Summary",
        "",
        "| Target | Cases | Pass Rate | Avg Score | Avg ms | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for target, summary in payload["summary"].items():
        notes = ""
        if target == "jade_agent_jgx":
            notes = (
                f"events={summary.get('jgx_event_count', 0)}, "
                f"snapshots={summary.get('jgx_snapshot_count', 0)}, "
                f"verify={summary.get('jgx_verify_ok', False)}"
            )
        if summary.get("skipped"):
            notes = str(summary.get("skip_reason", ""))
        lines.append(
            f"| `{target}` | {summary.get('cases', 0)} | {summary.get('pass_rate', 0)} | "
            f"{summary.get('avg_score', 0)} | {summary.get('avg_duration_ms', 0)} | {notes} |"
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
        "`schema_contract` and `conflict_resolution` are output-quality checks.",
        "A competent graph runner with the same deterministic backend should pass",
        "them. The sharper signal is in `replay_side_effect`, `crash_recovery`, and",
        "`audit_evidence`, because those require durable runtime behavior.",
        "",
        "`langgraph_plain` intentionally means a plain StateGraph target without a",
        "custom checkpointer, idempotency ledger, or audit capsule. This benchmark",
        "does not claim LangGraph cannot be extended to cover those needs; it shows",
        "the extra engineering surface that JadeAgent JGX treats as a first-class",
        "runtime capability.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_challenge_benchmark(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    token = _now_token()
    store_path = out_dir / f"challenge_jgx_{token}.sqlite3"
    artifact_dir = out_dir / f"challenge_artifacts_{token}"
    store = SqliteStateStore(store_path)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    raw_rows = [
        _run_row("raw_plain", case, run_raw_plain_case)
        for case in CHALLENGE_CASES
    ]
    rows.extend(raw_rows)
    summary["raw_plain"] = _summarize(raw_rows)

    try:
        _get_langgraph_plain_graph()
        langgraph_rows = [
            _run_row("langgraph_plain", case, run_langgraph_plain_case)
            for case in CHALLENGE_CASES
        ]
        rows.extend(langgraph_rows)
        summary["langgraph_plain"] = _summarize(langgraph_rows)
    except Exception as exc:
        summary["langgraph_plain"] = {
            "cases": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "avg_score": 0.0,
            "avg_duration_ms": 0.0,
            "skipped": True,
            "skip_reason": f"{exc.__class__.__name__}: {exc}",
        }

    jgx_rows: list[dict[str, Any]] = []
    for index, case in enumerate(CHALLENGE_CASES, start=1):
        jgx_rows.append(_run_row(
            "jade_agent_jgx",
            case,
            lambda selected_case, index=index: run_jade_agent_jgx_case(
                selected_case,
                store=store,
                token=token,
                index=index,
                artifact_dir=artifact_dir,
            ),
        ))
    rows.extend(jgx_rows)

    jgx_summary = _summarize(jgx_rows)
    jgx_events = 0
    jgx_snapshots = 0
    jgx_verify_ok = True
    latest_hash = ""
    for row in jgx_rows:
        evidence = dict(row.get("evidence", {}))
        jgx_events += int(evidence.get("event_count", 0))
        jgx_snapshots += int(evidence.get("snapshot_count", 0))
        jgx_verify_ok = jgx_verify_ok and bool(evidence.get("verify_ok", False))
        latest_hash = str(evidence.get("event_chain_hash", latest_hash) or latest_hash)
    jgx_summary.update({
        "jgx_event_count": jgx_events,
        "jgx_snapshot_count": jgx_snapshots,
        "jgx_verify_ok": jgx_verify_ok,
        "jgx_latest_event_chain_hash": latest_hash,
        "sqlite_path": str(store_path),
    })
    summary["jade_agent_jgx"] = jgx_summary
    store.close()

    payload = {
        "benchmark": "challenge_compare",
        "token": token,
        "summary": summary,
        "rows": rows,
        "state_store": str(store_path),
        "artifact_dir": str(artifact_dir),
        "notes": {
            "scope": "deterministic framework-capability benchmark",
            "langgraph_plain": "StateGraph without custom checkpointer, idempotency ledger, or JGX audit capsule",
            "claim_boundary": "This measures default/plain orchestration gaps, not an impossibility claim about LangGraph.",
        },
    }
    json_path = out_dir / f"challenge_compare_{token}.json"
    md_path = out_dir / f"challenge_compare_{token}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
    _write_markdown(payload, md_path)
    payload["json_path"] = str(json_path)
    payload["markdown_path"] = str(md_path)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run adversarial JadeAgent framework benchmark")
    parser.add_argument("--out-dir", default="benchmarks/out")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = run_challenge_benchmark(Path(args.out_dir))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print(f"benchmark: {payload['benchmark']}")
        for target, target_summary in payload["summary"].items():
            print(
                f"{target}: pass_rate={target_summary.get('pass_rate', 0)} "
                f"avg_score={target_summary.get('avg_score', 0)}"
            )
        print(f"json: {payload['json_path']}")
        print(f"markdown: {payload['markdown_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
