"""Portfolio-ready benchmark overview for JadeAgent.

This runner consolidates the deterministic local benchmark suite into one
report:

- runtime overhead: raw Python vs JadeGraph/JGX vs LangGraph when installed;
- controlled output quality: objective rubric with the same scripted backend;
- adversarial capability: replay, crash recovery, and audit evidence;
- JGX reliability eval: restore, idempotency, crash recovery, compatibility,
  raw-call overhead, and mesh project generation.

Run:

    python benchmarks/portfolio_overview.py --out-dir benchmarks/out --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.challenge_compare import run_challenge_benchmark
from benchmarks.durable_compare import run_durable_benchmark
from benchmarks.langgraph_compare import run_benchmark as run_runtime_benchmark
from benchmarks.quality_compare import run_quality_benchmark
from jadeagent.eval import run_eval_suite, write_markdown_report
from jadeagent.state import SqliteStateStore


def _now_token() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def _target(summary: dict[str, Any], name: str) -> dict[str, Any]:
    value = summary.get(name, {})
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _runtime_overview(runtime_payload: dict[str, Any]) -> dict[str, Any]:
    summary = runtime_payload.get("summary", {})
    raw = _target(summary, "raw_python")
    jade_graph = _target(summary, "jade_graph")
    jade_jgx = _target(summary, "jade_graph_jgx")
    langgraph = _target(summary, "langgraph")
    raw_p50 = _safe_float(raw.get("duration_ms_p50"))

    def overhead(target: dict[str, Any]) -> float:
        return round(_safe_float(target.get("duration_ms_p50")) - raw_p50, 6) if raw_p50 else 0.0

    return {
        "runs": runtime_payload.get("runs", 0),
        "raw_python_p50_ms": raw.get("duration_ms_p50", 0.0),
        "jade_graph_p50_ms": jade_graph.get("duration_ms_p50", 0.0),
        "jade_graph_overhead_vs_raw_p50_ms": overhead(jade_graph),
        "jade_graph_jgx_p50_ms": jade_jgx.get("duration_ms_p50", 0.0),
        "jade_graph_jgx_overhead_vs_raw_p50_ms": overhead(jade_jgx),
        "jade_graph_jgx_verify_ok": bool(jade_jgx.get("jgx_verify_ok", False)),
        "langgraph_p50_ms": langgraph.get("duration_ms_p50", 0.0),
        "langgraph_overhead_vs_raw_p50_ms": overhead(langgraph) if not langgraph.get("skipped") else None,
        "langgraph_skipped": bool(langgraph.get("skipped", False)),
        "langgraph_skip_reason": langgraph.get("skip_reason", ""),
    }


def _quality_overview(quality_payload: dict[str, Any]) -> dict[str, Any]:
    summary = quality_payload.get("summary", {})
    targets = {}
    for name in ("raw_baseline", "jade_agent", "langgraph", "jade_agent_jgx"):
        row = _target(summary, name)
        targets[name] = {
            "pass_rate": row.get("pass_rate", 0.0),
            "avg_score": row.get("avg_score", 0.0),
            "avg_duration_ms": row.get("avg_duration_ms", 0.0),
            "skipped": bool(row.get("skipped", False)),
        }
    return {
        "targets": targets,
        "jgx_verify_ok": bool(_target(summary, "jade_agent_jgx").get("jgx_verify_ok", False)),
        "claim": "Controlled backend quality parity is expected; frameworks do not improve the scripted answer by themselves.",
    }


def _challenge_overview(challenge_payload: dict[str, Any]) -> dict[str, Any]:
    summary = challenge_payload.get("summary", {})
    return {
        "raw_plain_pass_rate": _target(summary, "raw_plain").get("pass_rate", 0.0),
        "langgraph_plain_pass_rate": _target(summary, "langgraph_plain").get("pass_rate", 0.0),
        "jade_agent_jgx_pass_rate": _target(summary, "jade_agent_jgx").get("pass_rate", 0.0),
        "jade_agent_jgx_avg_score": _target(summary, "jade_agent_jgx").get("avg_score", 0.0),
        "jgx_event_count": _target(summary, "jade_agent_jgx").get("jgx_event_count", 0),
        "jgx_snapshot_count": _target(summary, "jade_agent_jgx").get("jgx_snapshot_count", 0),
        "jgx_verify_ok": bool(_target(summary, "jade_agent_jgx").get("jgx_verify_ok", False)),
    }


def _durable_overview(durable_payload: dict[str, Any]) -> dict[str, Any]:
    summary = durable_payload.get("summary", {})
    jgx = _target(summary, "jade_agent_jgx")
    langgraph = _target(summary, "langgraph_durable_sqlite")
    return {
        "jade_agent_jgx_pass_rate": jgx.get("pass_rate", 0.0),
        "jade_agent_jgx_recovery_pass_rate": jgx.get("recovery_pass_rate", 0.0),
        "jade_agent_jgx_avg_score": jgx.get("avg_score", 0.0),
        "jade_agent_jgx_verify_ok": bool(jgx.get("verify_ok", False)),
        "langgraph_durable_pass_rate": langgraph.get("pass_rate", 0.0),
        "langgraph_durable_recovery_pass_rate": langgraph.get("recovery_pass_rate", 0.0),
        "langgraph_durable_avg_score": langgraph.get("avg_score", 0.0),
        "langgraph_durable_skipped": bool(langgraph.get("skipped", False)),
        "langgraph_durable_skip_reason": langgraph.get("skip_reason", ""),
    }


def _eval_overview(eval_payload: dict[str, Any]) -> dict[str, Any]:
    aggregate = dict(eval_payload.get("aggregate", {}))
    return {
        "case_runs": aggregate.get("case_runs", 0),
        "success_rate": aggregate.get("success_rate", 0.0),
        "task_completion_rate": aggregate.get("task_completion_rate", 0.0),
        "recovery_success_rate": aggregate.get("recovery_success_rate", 0.0),
        "duplicate_tool_executions": aggregate.get("duplicate_tool_executions", 0),
        "secret_leak_count": aggregate.get("secret_leak_count", 0),
        "total_tokens": aggregate.get("total_tokens", 0),
        "runtime_ms_p50": aggregate.get("runtime_ms_p50", 0.0),
        "runtime_ms_p95": aggregate.get("runtime_ms_p95", 0.0),
        "restore_latency_ms_p50": aggregate.get("restore_latency_ms_p50", 0.0),
        "raw_runtime_ms_p50": aggregate.get("raw_runtime_ms_p50", 0.0),
        "jade_runtime_ms_p50": aggregate.get("jade_runtime_ms_p50", 0.0),
        "state_overhead_ms_p50": aggregate.get("state_overhead_ms_p50", 0.0),
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> Path:
    runtime = payload["overview"]["runtime"]
    quality = payload["overview"]["quality"]
    challenge = payload["overview"]["challenge"]
    durable = payload["overview"]["durable"]
    eval_overview = payload["overview"]["jgx_eval"]

    quality_targets = quality["targets"]
    lines = [
        "# JadeAgent Benchmark Overview",
        "",
        "Consolidated deterministic benchmark report for JadeAgent/JGX.",
        "",
        "## Executive Read",
        "",
        "- JadeGraph has very low local graph overhead compared with raw Python.",
        "- JGX adds measurable latency because it writes durable governed state.",
        "- Controlled output quality is expected to tie when the backend output is held constant.",
        "- JGX differentiates on reliability capabilities: idempotent replay, crash recovery, and audit evidence.",
        "",
        "## Runtime",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Runtime runs | {runtime['runs']} |",
        f"| Raw Python p50 ms | {runtime['raw_python_p50_ms']} |",
        f"| JadeGraph p50 ms | {runtime['jade_graph_p50_ms']} |",
        f"| JadeGraph overhead vs raw p50 ms | {runtime['jade_graph_overhead_vs_raw_p50_ms']} |",
        f"| JadeGraph+JGX p50 ms | {runtime['jade_graph_jgx_p50_ms']} |",
        f"| JadeGraph+JGX overhead vs raw p50 ms | {runtime['jade_graph_jgx_overhead_vs_raw_p50_ms']} |",
        f"| LangGraph p50 ms | {runtime['langgraph_p50_ms']} |",
        f"| JGX runtime integrity verified | {runtime['jade_graph_jgx_verify_ok']} |",
        "",
        "## Controlled Quality",
        "",
        "| Target | Pass Rate | Avg Score | Avg ms |",
        "|---|---:|---:|---:|",
    ]
    for target, row in quality_targets.items():
        lines.append(
            f"| `{target}` | {row['pass_rate']} | {row['avg_score']} | {row['avg_duration_ms']} |"
        )
    lines.extend([
        "",
        "## Adversarial Capability",
        "",
        "| Target | Pass Rate | Notes |",
        "|---|---:|---|",
        f"| `raw_plain` | {challenge['raw_plain_pass_rate']} | no durable replay/recovery/audit layer |",
        f"| `langgraph_plain` | {challenge['langgraph_plain_pass_rate']} | plain StateGraph baseline |",
        f"| `jade_agent_jgx` | {challenge['jade_agent_jgx_pass_rate']} | events={challenge['jgx_event_count']}, snapshots={challenge['jgx_snapshot_count']}, verify={challenge['jgx_verify_ok']} |",
        "",
        "## Durable LangGraph Comparison",
        "",
        "| Target | Pass Rate | Recovery Pass Rate | Avg Score | Notes |",
        "|---|---:|---:|---:|---|",
        f"| `jade_agent_jgx` | {durable['jade_agent_jgx_pass_rate']} | {durable['jade_agent_jgx_recovery_pass_rate']} | {durable['jade_agent_jgx_avg_score']} | verify={durable['jade_agent_jgx_verify_ok']} |",
        f"| `langgraph_durable_sqlite` | {durable['langgraph_durable_pass_rate']} | {durable['langgraph_durable_recovery_pass_rate']} | {durable['langgraph_durable_avg_score']} | production-style SQLite checkpointer |",
        "",
        "## JGX Reliability Eval",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Case runs | {eval_overview['case_runs']} |",
        f"| Success rate | {eval_overview['success_rate']} |",
        f"| Task completion rate | {eval_overview['task_completion_rate']} |",
        f"| Recovery success rate | {eval_overview['recovery_success_rate']} |",
        f"| Duplicate tool executions | {eval_overview['duplicate_tool_executions']} |",
        f"| Secret leak count | {eval_overview['secret_leak_count']} |",
        f"| Total tokens estimated | {eval_overview['total_tokens']} |",
        f"| Runtime p50/p95 ms | {eval_overview['runtime_ms_p50']} / {eval_overview['runtime_ms_p95']} |",
        f"| Restore latency p50 ms | {eval_overview['restore_latency_ms_p50']} |",
        f"| Raw vs Jade+JGX p50 ms | {eval_overview['raw_runtime_ms_p50']} / {eval_overview['jade_runtime_ms_p50']} |",
        f"| State overhead p50 ms | {eval_overview['state_overhead_ms_p50']} |",
        "",
        "## Artifact Map",
        "",
        f"- Runtime report: `{payload['artifacts']['runtime_markdown']}`",
        f"- Quality report: `{payload['artifacts']['quality_markdown']}`",
        f"- Challenge report: `{payload['artifacts']['challenge_markdown']}`",
        f"- Durable comparison report: `{payload['artifacts']['durable_markdown']}`",
        f"- Eval report: `{payload['artifacts']['eval_markdown']}`",
        "",
        "## Claim Boundary",
        "",
        "This suite is deterministic and local. It is strong for runtime capability,",
        "reliability, state integrity, and reproducibility claims. It is not a live",
        "LLM intelligence leaderboard. For that, add real model runs and an external",
        "judge or hidden task suite.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_portfolio_overview(
    out_dir: Path,
    *,
    runtime_runs: int = 25,
    eval_runs: int = 1,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    token = _now_token()
    root = out_dir / f"portfolio_overview_{token}"
    root.mkdir(parents=True, exist_ok=True)

    runtime_payload = run_runtime_benchmark(max(1, int(runtime_runs)), root / "runtime")
    quality_payload = run_quality_benchmark(root / "quality")
    challenge_payload = run_challenge_benchmark(root / "challenge")
    durable_payload = run_durable_benchmark(root / "durable")

    eval_store_path = root / f"eval_core_{token}.sqlite3"
    eval_store = SqliteStateStore(eval_store_path)
    try:
        eval_payload = run_eval_suite(
            eval_store,
            suite="core",
            runs=max(1, int(eval_runs)),
            output_dir=root / "eval",
        )
    finally:
        eval_store.close()
    eval_md_path = write_markdown_report(eval_payload, root / f"eval_core_{token}.md")

    artifacts = {
        "runtime_json": runtime_payload["json_path"],
        "runtime_markdown": runtime_payload["markdown_path"],
        "quality_json": quality_payload["json_path"],
        "quality_markdown": quality_payload["markdown_path"],
        "challenge_json": challenge_payload["json_path"],
        "challenge_markdown": challenge_payload["markdown_path"],
        "durable_json": durable_payload["json_path"],
        "durable_markdown": durable_payload["markdown_path"],
        "eval_store": str(eval_store_path),
        "eval_markdown": str(eval_md_path),
        "eval_output_dir": str(root / "eval"),
    }
    payload = {
        "benchmark": "portfolio_overview",
        "token": token,
        "root": str(root),
        "overview": {
            "runtime": _runtime_overview(runtime_payload),
            "quality": _quality_overview(quality_payload),
            "challenge": _challenge_overview(challenge_payload),
            "durable": _durable_overview(durable_payload),
            "jgx_eval": _eval_overview(eval_payload),
        },
        "artifacts": artifacts,
        "raw": {
            "runtime_summary": runtime_payload.get("summary", {}),
            "quality_summary": quality_payload.get("summary", {}),
            "challenge_summary": challenge_payload.get("summary", {}),
            "durable_summary": durable_payload.get("summary", {}),
            "eval_aggregate": eval_payload.get("aggregate", {}),
        },
        "notes": {
            "scope": "deterministic local overview",
            "claim_boundary": "Strong for runtime capability and reliability; not a live LLM intelligence leaderboard.",
        },
    }
    json_path = root / f"portfolio_overview_{token}.json"
    md_path = root / f"portfolio_overview_{token}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
    _write_markdown(payload, md_path)
    payload["json_path"] = str(json_path)
    payload["markdown_path"] = str(md_path)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run consolidated JadeAgent benchmark overview")
    parser.add_argument("--out-dir", default="benchmarks/out")
    parser.add_argument("--runtime-runs", type=int, default=25)
    parser.add_argument("--eval-runs", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = run_portfolio_overview(
        Path(args.out_dir),
        runtime_runs=max(1, int(args.runtime_runs)),
        eval_runs=max(1, int(args.eval_runs)),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print(f"benchmark: {payload['benchmark']}")
        print(f"root: {payload['root']}")
        print(f"markdown: {payload['markdown_path']}")
        print(f"json: {payload['json_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
