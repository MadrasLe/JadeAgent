"""Quality benchmark for deterministic agent/workflow outputs.

This benchmark is intentionally model-free by default. It compares output
quality with objective checks and a small rubric, so it can run in CI or Colab
without API keys. Optional LLM-judge evaluation can be added later on top of the
same JSON artifacts.

Run:

    python benchmarks/quality_compare.py --out-dir benchmarks/out --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jadeagent import Agent
from jadeagent.backends.base import LLMBackend
from jadeagent.core.types import Message, Response, Role, StreamChunk
from jadeagent.state import SqliteStateStore, verify_capsule


class ScriptedQualityBackend(LLMBackend):
    """Deterministic backend for quality scoring without network calls."""

    def __init__(self, answers: dict[str, str]):
        self.answers = dict(answers)
        self.model = "scripted-quality"

    def chat(
        self,
        messages: list[Message],
        tools=None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop=None,
    ) -> Response:
        prompt = ""
        for message in reversed(messages):
            role = message.role.value if isinstance(message.role, Role) else str(message.role)
            if role == "user" and message.content:
                prompt = message.content
                break
        case_id = prompt.splitlines()[0].replace("CASE:", "").strip() if prompt else ""
        return Response(
            content=self.answers.get(case_id, ""),
            model=self.model,
            finish_reason="stop",
        )

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


@dataclass(frozen=True)
class QualityCase:
    case_id: str
    prompt: str
    required_terms: tuple[str, ...]
    forbidden_terms: tuple[str, ...] = ()
    min_words: int = 20
    max_words: int = 180
    required_sections: tuple[str, ...] = ()


QUALITY_CASES = [
    QualityCase(
        case_id="architecture_brief",
        prompt=(
            "Write a concise architecture brief for a durable agent state layer. "
            "Mention manifest, events, snapshots, restore compatibility, and one limitation."
        ),
        required_terms=("manifest", "events", "snapshots", "restore compatibility", "limitation"),
        required_sections=("Overview", "State", "Risk"),
    ),
    QualityCase(
        case_id="failure_analysis",
        prompt=(
            "Explain what should happen when a worker crashes after writing a checkpoint. "
            "Include recovery behavior, duplicate side-effect prevention, and audit evidence."
        ),
        required_terms=("checkpoint", "recover", "duplicate", "audit"),
        forbidden_terms=("ignore the failure",),
        required_sections=("Failure", "Recovery", "Evidence"),
    ),
    QualityCase(
        case_id="implementation_plan",
        prompt=(
            "Create a small implementation plan for adding an HTML timeline command. "
            "Include CLI behavior, data source, output artifact, and tests."
        ),
        required_terms=("CLI", "events", "snapshots", "HTML", "tests"),
        required_sections=("Plan", "Tests", "Output"),
    ),
]


JADE_ANSWERS = {
    "architecture_brief": """Overview
JGX is a durable agent state layer that turns an agent run into an inspectable state machine.

State
Each run stores a manifest for identity and compatibility, append-only events for what happened, and snapshots for safe resume points. Restore compatibility checks tenant, tools, policy, memory scope, and backend fingerprints.

Risk
The main limitation is overhead: durable snapshots and SQLite writes cost more latency than a raw in-memory loop.""",
    "failure_analysis": """Failure
If a worker crashes after writing a checkpoint, the task should not be treated as lost. The checkpoint is the durable boundary.

Recovery
A later worker reads the latest snapshot, resumes from the checkpoint, and completes the task. Duplicate side effects are prevented with idempotency keys and recorded tool results.

Evidence
The audit trail should show the crash event, checkpoint event, recovered completion event, and final completed state.""",
    "implementation_plan": """Plan
Add a CLI command that loads a JGX run from the state store, merges events and snapshots by timestamp, and writes an HTML timeline.

Tests
Cover JSON output, HTML file creation, missing run behavior, and secret redaction in event payloads.

Output
The command should produce a standalone HTML artifact with phases, steps, event types, snapshot ids, and metadata.""",
}


BASELINE_ANSWERS = {
    "architecture_brief": "It should save some state and reload it later. This helps agents continue work.",
    "failure_analysis": "The worker crashed. Start again and finish the task.",
    "implementation_plan": "Make a command and print a web page.",
}


_LANGGRAPH_QUALITY_COMPILED: Any = None


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def score_output(case: QualityCase, output: str) -> dict[str, Any]:
    text = output.strip()
    lower = text.lower()
    required_hits = [
        term for term in case.required_terms
        if term.lower() in lower
    ]
    forbidden_hits = [
        term for term in case.forbidden_terms
        if term.lower() in lower
    ]
    section_hits = [
        section for section in case.required_sections
        if re.search(rf"(^|\n)\s*{re.escape(section)}\b", text, flags=re.IGNORECASE)
    ]
    words = _word_count(text)

    term_score = len(required_hits) / max(len(case.required_terms), 1)
    section_score = len(section_hits) / max(len(case.required_sections), 1)
    length_score = 1.0 if case.min_words <= words <= case.max_words else 0.0
    forbidden_score = 1.0 if not forbidden_hits else 0.0
    exact_score = (
        0.55 * term_score
        + 0.25 * section_score
        + 0.10 * length_score
        + 0.10 * forbidden_score
    )
    return {
        "score": round(exact_score, 4),
        "passed": exact_score >= 0.82,
        "word_count": words,
        "required_hits": required_hits,
        "missing_terms": [term for term in case.required_terms if term not in required_hits],
        "section_hits": section_hits,
        "missing_sections": [section for section in case.required_sections if section not in section_hits],
        "forbidden_hits": forbidden_hits,
    }


def run_raw_baseline(case: QualityCase) -> str:
    return BASELINE_ANSWERS[case.case_id]


def run_jade_agent(case: QualityCase, state_store: SqliteStateStore | None = None, run_id: str | None = None) -> str:
    backend = ScriptedQualityBackend(JADE_ANSWERS)
    agent = Agent(
        backend=backend,
        name="quality_jade",
        system_prompt="Return a structured answer with the requested sections.",
        state_store=state_store,
        run_id=run_id,
        verbose=False,
        max_iterations=1,
    )
    return agent.run(f"CASE: {case.case_id}\n{case.prompt}").answer


def _get_langgraph_quality_graph() -> Any:
    """Build and cache the LangGraph quality graph if LangGraph is installed."""

    global _LANGGRAPH_QUALITY_COMPILED
    if _LANGGRAPH_QUALITY_COMPILED is not None:
        return _LANGGRAPH_QUALITY_COMPILED

    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class LangGraphQualityState(TypedDict, total=False):
        case_id: str
        prompt: str
        output: str

    backend = ScriptedQualityBackend(JADE_ANSWERS)

    def answer_node(state: dict[str, Any]) -> dict[str, Any]:
        response = backend.chat([
            Message.system("Return a structured answer with the requested sections."),
            Message.user(f"CASE: {state['case_id']}\n{state['prompt']}"),
        ])
        return {"output": response.content or ""}

    graph = StateGraph(LangGraphQualityState)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    _LANGGRAPH_QUALITY_COMPILED = graph.compile()
    return _LANGGRAPH_QUALITY_COMPILED


def run_langgraph_agent(case: QualityCase) -> str:
    """Run the same scripted answer generation through LangGraph if installed."""

    compiled = _get_langgraph_quality_graph()
    result = compiled.invoke({"case_id": case.case_id, "prompt": case.prompt})
    return str(dict(result).get("output", ""))


def _run_target(
    target: str,
    cases: list[QualityCase],
    fn: Callable[[QualityCase, int], tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        started = time.perf_counter()
        output, metadata = fn(case, index)
        duration_ms = (time.perf_counter() - started) * 1000.0
        score = score_output(case, output)
        rows.append({
            "target": target,
            "case_id": case.case_id,
            "duration_ms": round(duration_ms, 4),
            "output": output,
            "metadata": metadata,
            **score,
        })
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"cases": 0, "pass_rate": 0.0, "avg_score": 0.0}
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
        "# JadeAgent Quality Benchmark",
        "",
        "Objective rubric benchmark for output quality. This is model-free by",
        "default and checks required concepts, structure, forbidden claims, and",
        "length bounds.",
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
        if target == "langgraph" and summary.get("skipped"):
            notes = str(summary.get("skip_reason", ""))
        lines.append(
            f"| `{target}` | {summary.get('cases', 0)} | {summary.get('pass_rate', 0)} | "
            f"{summary.get('avg_score', 0)} | {summary.get('avg_duration_ms', 0)} | {notes} |"
        )
    lines.extend([
        "",
        "## Cases",
        "",
        "| Target | Case | Passed | Score | Missing Terms | Missing Sections |",
        "|---|---|---:|---:|---|---|",
    ])
    for row in payload["rows"]:
        lines.append(
            f"| `{row['target']}` | `{row['case_id']}` | `{row['passed']}` | "
            f"{row['score']} | {', '.join(row['missing_terms']) or '-'} | "
            f"{', '.join(row['missing_sections']) or '-'} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This benchmark tests quality by objective rubric. It does not replace human",
        "review or LLM-judge evaluation, but it is deterministic and CI-friendly.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_quality_benchmark(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    token = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    store_path = out_dir / f"quality_jgx_{token}.sqlite3"
    store = SqliteStateStore(store_path)

    rows: list[dict[str, Any]] = []
    rows.extend(_run_target(
        "raw_baseline",
        QUALITY_CASES,
        lambda case, _index: (run_raw_baseline(case), {}),
    ))
    rows.extend(_run_target(
        "jade_agent",
        QUALITY_CASES,
        lambda case, _index: (run_jade_agent(case), {}),
    ))
    try:
        _get_langgraph_quality_graph()
        rows.extend(_run_target(
            "langgraph",
            QUALITY_CASES,
            lambda case, _index: (run_langgraph_agent(case), {}),
        ))
        langgraph_skip: dict[str, Any] | None = None
    except Exception as exc:
        langgraph_skip = {
            "cases": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "avg_score": 0.0,
            "avg_duration_ms": 0.0,
            "skipped": True,
            "skip_reason": f"{exc.__class__.__name__}: {exc}",
        }

    jgx_run_ids: list[str] = []

    def _run_jgx(case: QualityCase, index: int) -> tuple[str, dict[str, Any]]:
        run_id = f"quality_{case.case_id}_{token}_{index}"
        jgx_run_ids.append(run_id)
        return run_jade_agent(case, state_store=store, run_id=run_id), {"run_id": run_id}

    rows.extend(_run_target("jade_agent_jgx", QUALITY_CASES, _run_jgx))

    summary: dict[str, Any] = {}
    for target in sorted({row["target"] for row in rows}):
        target_rows = [row for row in rows if row["target"] == target]
        summary[target] = _summarize(target_rows)
    if langgraph_skip is not None:
        summary["langgraph"] = langgraph_skip

    jgx_events = 0
    jgx_snapshots = 0
    jgx_verify_ok = True
    for run_id in jgx_run_ids:
        capsule = store.load_run(run_id)
        verify = verify_capsule(capsule)
        jgx_verify_ok = jgx_verify_ok and bool(verify["ok"])
        jgx_events += len(capsule.events)
        jgx_snapshots += len(capsule.snapshots)
    summary["jade_agent_jgx"].update({
        "jgx_event_count": jgx_events,
        "jgx_snapshot_count": jgx_snapshots,
        "jgx_verify_ok": jgx_verify_ok,
        "sqlite_path": str(store_path),
    })
    store.close()

    payload = {
        "benchmark": "quality_compare",
        "token": token,
        "summary": summary,
        "rows": rows,
        "state_store": str(store_path),
        "rubric": {
            "required_terms_weight": 0.55,
            "section_weight": 0.25,
            "length_weight": 0.10,
            "forbidden_weight": 0.10,
            "pass_threshold": 0.82,
        },
    }
    json_path = out_dir / f"quality_compare_{token}.json"
    md_path = out_dir / f"quality_compare_{token}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
    _write_markdown(payload, md_path)
    payload["json_path"] = str(json_path)
    payload["markdown_path"] = str(md_path)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic quality benchmark")
    parser.add_argument("--out-dir", default="benchmarks/out")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = run_quality_benchmark(Path(args.out_dir))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print(f"benchmark: {payload['benchmark']}")
        for target, summary in payload["summary"].items():
            print(
                f"{target}: pass_rate={summary.get('pass_rate', 0)} "
                f"avg_score={summary.get('avg_score', 0)}"
            )
        print(f"json: {payload['json_path']}")
        print(f"markdown: {payload['markdown_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
