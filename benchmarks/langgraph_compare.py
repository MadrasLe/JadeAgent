"""Colab-friendly benchmark: raw Python vs JadeGraph/JGX vs LangGraph.

The benchmark is deterministic on purpose. It measures runtime/framework
overhead and state/audit capability without mixing in network latency or model
quality. In Colab, install LangGraph first to enable the LangGraph target:

    pip install -q langgraph

Run:

    python benchmarks/langgraph_compare.py --runs 25 --out-dir benchmarks/out
"""

from __future__ import annotations

import argparse
import json
import operator
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, TypedDict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jadeagent.graph import END as JADE_END
from jadeagent.graph import START as JADE_START
from jadeagent.graph import StateGraph as JadeStateGraph
from jadeagent.state import SqliteStateStore, verify_capsule


class BenchState(TypedDict, total=False):
    prompt: str
    normalized: str
    plan: list[str]
    artifact: str
    tests_passed: bool
    report: str
    steps: Annotated[list[str], operator.add]


def _step(state: dict[str, Any], name: str, updates: dict[str, Any]) -> dict[str, Any]:
    return {**updates, "steps": [name]}


def normalize_node(state: dict[str, Any]) -> dict[str, Any]:
    normalized = " ".join(str(state.get("prompt", "")).lower().split())
    return _step(state, "normalize", {"normalized": normalized})


def plan_node(state: dict[str, Any]) -> dict[str, Any]:
    topic = str(state.get("normalized", "task"))
    plan = [
        f"define interface for {topic}",
        "implement deterministic planner",
        "validate dependency order",
        "emit concise report",
    ]
    return _step(state, "plan", {"plan": plan})


def implement_node(state: dict[str, Any]) -> dict[str, Any]:
    lines = [
        "# generated taskpack",
        *[f"- {item}" for item in state.get("plan", [])],
    ]
    return _step(state, "implement", {"artifact": "\n".join(lines)})


def test_node(state: dict[str, Any]) -> dict[str, Any]:
    artifact = str(state.get("artifact", ""))
    plan = list(state.get("plan", []))
    tests_passed = (
        artifact.startswith("# generated taskpack")
        and len(plan) == 4
        and "validate dependency order" in artifact
    )
    return _step(state, "test", {"tests_passed": tests_passed})


def report_node(state: dict[str, Any]) -> dict[str, Any]:
    status = "completed" if state.get("tests_passed") else "failed"
    report = (
        f"status={status}; "
        f"steps={len(state.get('steps', [])) + 1}; "
        f"artifact_chars={len(str(state.get('artifact', '')))}"
    )
    return _step(state, "report", {"report": report})


NODES = [normalize_node, plan_node, implement_node, test_node, report_node]
NODE_NAMES = ["normalize", "plan", "implement", "test", "report"]


def initial_state() -> dict[str, Any]:
    return {"prompt": "Build a tiny dependency-aware task planner", "steps": []}


def run_raw() -> dict[str, Any]:
    state = initial_state()
    for fn in NODES:
        update = fn(state)
        for key, value in update.items():
            if key in state and isinstance(state[key], list) and isinstance(value, list):
                state[key] = state[key] + value
            else:
                state[key] = value
    return state


def build_jade_graph():
    graph = JadeStateGraph()
    for name, fn in zip(NODE_NAMES, NODES):
        graph.add_node(name, fn)
    graph.add_edge(JADE_START, "normalize")
    graph.add_edge("normalize", "plan")
    graph.add_edge("plan", "implement")
    graph.add_edge("implement", "test")
    graph.add_edge("test", "report")
    graph.add_edge("report", JADE_END)
    return graph.compile()


def run_jade_graph(compiled) -> dict[str, Any]:
    return compiled.run(initial_state())


def run_jade_graph_jgx(compiled, store: SqliteStateStore, run_id: str) -> dict[str, Any]:
    return compiled.run(initial_state(), state_store=store, run_id=run_id)


def build_langgraph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(BenchState)
    for name, fn in zip(NODE_NAMES, NODES):
        graph.add_node(name, fn)
    graph.add_edge(START, "normalize")
    graph.add_edge("normalize", "plan")
    graph.add_edge("plan", "implement")
    graph.add_edge("implement", "test")
    graph.add_edge("test", "report")
    graph.add_edge("report", END)
    return graph.compile()


def run_langgraph(compiled) -> dict[str, Any]:
    return dict(compiled.invoke(initial_state()))


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return float(ordered[index])


def _status(state: dict[str, Any]) -> str:
    return "completed" if state.get("tests_passed") and "status=completed" in str(state.get("report", "")) else "failed"


def _run_target(name: str, runs: int, fn) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    error = ""
    for index in range(1, runs + 1):
        started = time.perf_counter()
        try:
            state = fn(index)
            duration_ms = (time.perf_counter() - started) * 1000.0
            rows.append({
                "target": name,
                "iteration": index,
                "duration_ms": round(duration_ms, 6),
                "status": _status(state),
                "task_completed": _status(state) == "completed",
                "steps": len(state.get("steps", [])),
                "report": state.get("report", ""),
                "error": "",
            })
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            error = f"{exc.__class__.__name__}: {exc}"
            rows.append({
                "target": name,
                "iteration": index,
                "duration_ms": round(duration_ms, 6),
                "status": "failed",
                "task_completed": False,
                "steps": 0,
                "report": "",
                "error": error,
            })
    return rows, error


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(row["duration_ms"]) for row in rows]
    completed = [row for row in rows if row["task_completed"]]
    return {
        "runs": len(rows),
        "completed": len(completed),
        "failed": len(rows) - len(completed),
        "completion_rate": round(len(completed) / len(rows), 4) if rows else 0.0,
        "duration_ms_avg": round(statistics.mean(durations), 6) if durations else 0.0,
        "duration_ms_p50": round(_percentile(durations, 0.50), 6),
        "duration_ms_p95": round(_percentile(durations, 0.95), 6),
        "min_ms": round(min(durations), 6) if durations else 0.0,
        "max_ms": round(max(durations), 6) if durations else 0.0,
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> Path:
    lines = [
        "# LangGraph Comparison Benchmark",
        "",
        "Deterministic workflow benchmark comparing raw Python, JadeGraph,",
        "JadeGraph+JGX, and LangGraph when installed.",
        "",
        "## Summary",
        "",
        "| Target | Runs | Completion | p50 ms | p95 ms | Avg ms | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    raw_p50 = payload["summary"].get("raw_python", {}).get("duration_ms_p50", 0.0)
    for target, summary in payload["summary"].items():
        notes = ""
        if target == "jade_graph_jgx":
            notes = (
                f"events={summary.get('jgx_event_count', 0)}, "
                f"snapshots={summary.get('jgx_snapshot_count', 0)}, "
                f"verify={summary.get('jgx_verify_ok', False)}"
            )
        if target == "langgraph" and summary.get("skipped"):
            notes = summary.get("skip_reason", "")
        overhead = ""
        if raw_p50 and target != "raw_python":
            overhead = f"; overhead_vs_raw_p50={round(summary.get('duration_ms_p50', 0.0) - raw_p50, 6)}"
        lines.append(
            f"| `{target}` | {summary.get('runs', 0)} | {summary.get('completion_rate', 0.0)} | "
            f"{summary.get('duration_ms_p50', 0.0)} | {summary.get('duration_ms_p95', 0.0)} | "
            f"{summary.get('duration_ms_avg', 0.0)} | {notes}{overhead} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Raw Python is the lower-bound overhead. JadeGraph measures the local graph",
        "engine. JadeGraph+JGX measures the cost of durable governed state.",
        "LangGraph measures the external framework baseline for the same deterministic",
        "state transition graph.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_benchmark(runs: int, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    token = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    store_path = out_dir / f"jade_jgx_{token}.sqlite3"
    store = SqliteStateStore(store_path)
    jade_compiled = build_jade_graph()

    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    rows, _ = _run_target("raw_python", runs, lambda _index: run_raw())
    all_rows.extend(rows)
    summary["raw_python"] = _summarize(rows)

    rows, _ = _run_target("jade_graph", runs, lambda _index: run_jade_graph(jade_compiled))
    all_rows.extend(rows)
    summary["jade_graph"] = _summarize(rows)

    jgx_run_ids: list[str] = []

    def _jgx(index: int) -> dict[str, Any]:
        run_id = f"bench_jade_jgx_{token}_{index}"
        jgx_run_ids.append(run_id)
        return run_jade_graph_jgx(jade_compiled, store, run_id)

    rows, _ = _run_target("jade_graph_jgx", runs, _jgx)
    all_rows.extend(rows)
    jgx_summary = _summarize(rows)
    jgx_events = 0
    jgx_snapshots = 0
    verify_ok = True
    latest_verify: dict[str, Any] = {}
    for run_id in jgx_run_ids:
        capsule = store.load_run(run_id)
        verify = verify_capsule(capsule)
        latest_verify = verify
        verify_ok = verify_ok and bool(verify["ok"])
        jgx_events += len(capsule.events)
        jgx_snapshots += len(capsule.snapshots)
    jgx_summary.update({
        "jgx_event_count": jgx_events,
        "jgx_snapshot_count": jgx_snapshots,
        "jgx_verify_ok": verify_ok,
        "jgx_latest_event_chain_hash": latest_verify.get("event_chain_hash", ""),
        "sqlite_path": str(store_path),
    })
    summary["jade_graph_jgx"] = jgx_summary

    try:
        langgraph_compiled = build_langgraph()
        rows, _ = _run_target("langgraph", runs, lambda _index: run_langgraph(langgraph_compiled))
        all_rows.extend(rows)
        summary["langgraph"] = _summarize(rows)
    except Exception as exc:
        summary["langgraph"] = {
            "runs": 0,
            "completed": 0,
            "failed": 0,
            "completion_rate": 0.0,
            "duration_ms_avg": 0.0,
            "duration_ms_p50": 0.0,
            "duration_ms_p95": 0.0,
            "skipped": True,
            "skip_reason": f"{exc.__class__.__name__}: {exc}",
        }

    store.close()

    payload = {
        "benchmark": "langgraph_compare",
        "token": token,
        "runs": runs,
        "state_store": str(store_path),
        "summary": summary,
        "rows": all_rows,
        "notes": {
            "task": "deterministic five-node workflow",
            "langgraph_docs": [
                "https://docs.langchain.com/oss/python/langgraph/use-graph-api",
                "https://reference.langchain.com/python/langgraph/graph/state/StateGraph",
            ],
        },
    }
    json_path = out_dir / f"langgraph_compare_{token}.json"
    md_path = out_dir / f"langgraph_compare_{token}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
    _write_markdown(payload, md_path)
    payload["json_path"] = str(json_path)
    payload["markdown_path"] = str(md_path)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark JadeAgent against LangGraph")
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--out-dir", default="benchmarks/out")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = run_benchmark(max(1, int(args.runs)), Path(args.out_dir))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print(f"benchmark: {payload['benchmark']}")
        print(f"runs: {payload['runs']}")
        for target, summary in payload["summary"].items():
            print(
                f"{target}: completion={summary.get('completion_rate', 0)} "
                f"p50={summary.get('duration_ms_p50', 0)}ms "
                f"p95={summary.get('duration_ms_p95', 0)}ms"
            )
        print(f"json: {payload['json_path']}")
        print(f"markdown: {payload['markdown_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
