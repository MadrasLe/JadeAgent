"""Command line interface for JadeAgent."""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from .eval import build_eval_report_payload, run_eval_suite, write_markdown_report
from .state import FileStateStore, SqliteStateStore, StateStore
from .state.integrity import redact_secrets, verify_capsule


def _choose_store(path: str, store_type: str = "auto") -> StateStore:
    raw = Path(path)
    selected = store_type
    if selected == "auto":
        suffix = raw.suffix.lower()
        if suffix in {".db", ".sqlite", ".sqlite3"} or (raw.exists() and raw.is_file()):
            selected = "sqlite"
        else:
            selected = "file"

    if selected == "sqlite":
        return SqliteStateStore(raw)
    if selected == "file":
        return FileStateStore(raw)
    raise ValueError(f"unsupported store type: {store_type}")


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True))


def _format_ts(timestamp: float) -> str:
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")


def _close_store(store: StateStore) -> None:
    close = getattr(store, "close", None)
    if callable(close):
        close()


def _snapshot_summary(snapshot: Any) -> dict[str, Any]:
    if snapshot is None:
        return {}
    return {
        "snapshot_id": snapshot.snapshot_id,
        "phase": snapshot.phase,
        "step": snapshot.step,
        "created_at": snapshot.created_at,
        "message_count": len(snapshot.messages),
        "pending_tool": (
            snapshot.pending_tool_call.get("name")
            if isinstance(snapshot.pending_tool_call, dict)
            else ""
        ),
        "graph_node": snapshot.graph.current_node if snapshot.graph else "",
        "mesh_state": snapshot.mesh.task_state if snapshot.mesh else "",
        "metadata": dict(snapshot.metadata),
    }


def _state_inspect(args: argparse.Namespace) -> int:
    store = _choose_store(args.store, args.store_type)
    try:
        info = store.inspect(args.run_id)
        if args.json:
            _print_json(info)
        else:
            print(f"run_id: {info['run_id']}")
            print(f"format: {info['format']} ({info['magic']} schema {info['schema_version']})")
            print(f"agent_id: {info['agent_id']}")
            print(f"task_id: {info['task_id']}")
            print(f"tenant_id: {info['tenant_id']}")
            print(f"capability: {info['capability']}")
            print(f"backend: {info['backend']}")
            print(f"latest_phase: {info['latest_phase']}")
            print(f"latest_step: {info['latest_step']}")
            print(f"snapshots: {info['snapshot_count']}")
            print(f"events: {info['event_count']}")
            print(f"payloads: {info['payload_count']}")
        return 0
    except Exception as exc:
        print(f"jade state inspect failed: {exc}", file=sys.stderr)
        return 1
    finally:
        _close_store(store)


def _state_history(args: argparse.Namespace) -> int:
    store = _choose_store(args.store, args.store_type)
    try:
        events = store.list_events(args.run_id, limit=args.limit)
        if args.json:
            _print_json([event.to_dict() for event in events])
        else:
            if not events:
                print(f"no events for run_id: {args.run_id}")
                return 0
            print(f"history for {args.run_id} (last {len(events)} event(s))")
            for event in events:
                event_id = event.event_id[:10]
                ts = _format_ts(event.timestamp)
                details = f" {event.message}" if event.message else ""
                print(
                    f"{ts} [{event.step}] {event.phase or '-'} "
                    f"{event.event_type} {event_id}{details}"
                )
        return 0
    except Exception as exc:
        print(f"jade state history failed: {exc}", file=sys.stderr)
        return 1
    finally:
        _close_store(store)


def _state_list(args: argparse.Namespace) -> int:
    store = _choose_store(args.store, args.store_type)
    try:
        list_runs = getattr(store, "list_runs", None)
        if not callable(list_runs):
            raise ValueError("store does not support list_runs")
        run_ids = list_runs()
        if args.json:
            _print_json([store.inspect(run_id) for run_id in run_ids])
        else:
            if not run_ids:
                print("no state runs found")
                return 0
            print(f"{len(run_ids)} state run(s)")
            for run_id in run_ids:
                info = store.inspect(run_id)
                print(
                    f"{run_id}  {info['latest_phase'] or '-'}  "
                    f"snapshots={info['snapshot_count']} events={info['event_count']} "
                    f"agent={info['agent_id'] or '-'} capability={info['capability'] or '-'}"
                )
        return 0
    except Exception as exc:
        print(f"jade state list failed: {exc}", file=sys.stderr)
        return 1
    finally:
        _close_store(store)


def _state_latest(args: argparse.Namespace) -> int:
    store = _choose_store(args.store, args.store_type)
    try:
        snapshot = store.latest_snapshot(args.run_id)
        if snapshot is None:
            print(f"no snapshots for run_id: {args.run_id}", file=sys.stderr)
            return 1
        if args.json:
            _print_json(snapshot.to_dict())
        else:
            summary = _snapshot_summary(snapshot)
            print(f"snapshot_id: {summary['snapshot_id']}")
            print(f"phase: {summary['phase']}")
            print(f"step: {summary['step']}")
            print(f"created_at: {_format_ts(summary['created_at'])}")
            print(f"messages: {summary['message_count']}")
            print(f"pending_tool: {summary['pending_tool']}")
            print(f"graph_node: {summary['graph_node']}")
            print(f"mesh_state: {summary['mesh_state']}")
        return 0
    except Exception as exc:
        print(f"jade state latest failed: {exc}", file=sys.stderr)
        return 1
    finally:
        _close_store(store)


def _state_export(args: argparse.Namespace) -> int:
    store = _choose_store(args.store, args.store_type)
    try:
        destination = Path(args.out or f"{args.run_id}.jgx")
        export_run = getattr(store, "export_run", None)
        if callable(export_run):
            output = export_run(args.run_id, destination)
        else:
            output = store.load_run(args.run_id).to_directory(destination)
        if args.json:
            _print_json({"run_id": args.run_id, "output": str(output)})
        else:
            print(f"exported {args.run_id} -> {output}")
        return 0
    except Exception as exc:
        print(f"jade state export failed: {exc}", file=sys.stderr)
        return 1
    finally:
        _close_store(store)


def _timeline_items(capsule: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for event in capsule.events:
        items.append({
            "kind": "event",
            "timestamp": event.timestamp,
            "time": _format_ts(event.timestamp),
            "phase": event.phase,
            "step": event.step,
            "type": event.event_type,
            "actor": event.actor,
            "message": event.message,
            "details": redact_secrets(event.payload),
        })
    for snapshot in capsule.snapshots:
        details: dict[str, Any] = {
            "snapshot_id": snapshot.snapshot_id,
            "messages": len(snapshot.messages),
            "metadata": snapshot.metadata,
        }
        if snapshot.pending_tool_call:
            details["pending_tool"] = snapshot.pending_tool_call.get("name")
        if snapshot.graph:
            details["graph_node"] = snapshot.graph.current_node
        if snapshot.mesh:
            details["mesh_state"] = snapshot.mesh.task_state
        items.append({
            "kind": "snapshot",
            "timestamp": snapshot.created_at,
            "time": _format_ts(snapshot.created_at),
            "phase": snapshot.phase,
            "step": snapshot.step,
            "type": "snapshot",
            "actor": "",
            "message": "runtime snapshot saved",
            "details": redact_secrets(details),
        })
    return sorted(items, key=lambda item: (float(item.get("timestamp", 0.0)), item["kind"]))


def _write_timeline_html(capsule: Any, items: list[dict[str, Any]], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, item in enumerate(items, start=1):
        details = json.dumps(item.get("details", {}), indent=2, sort_keys=True, ensure_ascii=True)
        rows.append(
            "<article class='item'>"
            f"<div class='index'>{index}</div>"
            "<div class='body'>"
            f"<div class='meta'>{html.escape(item['time'])} | step {html.escape(str(item['step']))} | {html.escape(item['kind'])}</div>"
            f"<h2>{html.escape(item['phase'] or '-')} <span>{html.escape(item['type'])}</span></h2>"
            f"<p>{html.escape(item.get('message') or '')}</p>"
            f"<pre>{html.escape(details)}</pre>"
            "</div>"
            "</article>"
        )
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>JGX Timeline {html.escape(capsule.run_id)}</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; }}
    body {{ margin: 0; background: #f7f7f4; color: #20201d; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 32px 20px 56px; }}
    header {{ border-bottom: 1px solid #d9d5ca; margin-bottom: 24px; padding-bottom: 16px; }}
    h1 {{ font-size: 28px; margin: 0 0 8px; letter-spacing: 0; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 10px; font-size: 14px; color: #4d4b45; }}
    .pill {{ border: 1px solid #cbc6ba; border-radius: 6px; padding: 6px 8px; background: #fffdfa; }}
    .item {{ display: grid; grid-template-columns: 44px minmax(0, 1fr); gap: 14px; margin: 0 0 12px; }}
    .index {{ width: 32px; height: 32px; border-radius: 50%; background: #283618; color: white; display: grid; place-items: center; font-weight: 700; }}
    .body {{ background: #fffdfa; border: 1px solid #d9d5ca; border-radius: 8px; padding: 14px; }}
    .meta {{ color: #716d64; font-size: 13px; margin-bottom: 4px; }}
    h2 {{ font-size: 18px; margin: 0 0 6px; letter-spacing: 0; }}
    h2 span {{ color: #6b705c; font-size: 13px; font-weight: 600; }}
    p {{ margin: 0 0 10px; }}
    pre {{ overflow: auto; background: #272822; color: #f8f8f2; border-radius: 6px; padding: 10px; font-size: 12px; line-height: 1.45; }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>JGX Timeline</h1>
    <div class="summary">
      <span class="pill">run_id: {html.escape(capsule.run_id)}</span>
      <span class="pill">events: {len(capsule.events)}</span>
      <span class="pill">snapshots: {len(capsule.snapshots)}</span>
      <span class="pill">latest: {html.escape(capsule.latest_snapshot.phase if capsule.latest_snapshot else "")}</span>
    </div>
  </header>
  {''.join(rows)}
</main>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")
    return path


def _state_timeline(args: argparse.Namespace) -> int:
    store = _choose_store(args.store, args.store_type)
    try:
        capsule = store.load_run(args.run_id)
        items = _timeline_items(capsule)
        payload = {
            "run_id": args.run_id,
            "event_count": len(capsule.events),
            "snapshot_count": len(capsule.snapshots),
            "items": items,
        }
        if args.html:
            output = _write_timeline_html(capsule, items, args.html)
            payload["html"] = str(output)
        if args.json:
            _print_json(payload)
        else:
            if args.html:
                print(f"timeline html: {payload['html']}")
            print(f"timeline for {args.run_id} ({len(items)} item(s))")
            for item in items:
                print(
                    f"{item['time']} [{item['step']}] {item['kind']} "
                    f"{item['phase'] or '-'} {item['type']} {item.get('message') or ''}"
                )
        return 0
    except Exception as exc:
        print(f"jade state timeline failed: {exc}", file=sys.stderr)
        return 1
    finally:
        _close_store(store)


def _state_verify(args: argparse.Namespace) -> int:
    store = _choose_store(args.store, args.store_type)
    try:
        report = verify_capsule(store.load_run(args.run_id))
        if args.json:
            _print_json(report)
        else:
            print(f"run_id: {report['run_id']}")
            print(f"ok: {report['ok']}")
            print(f"event_chain_hash: {report['event_chain_hash']}")
            print(f"snapshot_hash: {report['snapshot_hash']}")
            print(f"secret_leak_count: {report['secret_leak_count']}")
            if report["issues"]:
                print("issues:")
                for issue in report["issues"]:
                    print(f"- {issue}")
        return 0 if report["ok"] else 1
    except Exception as exc:
        print(f"jade state verify failed: {exc}", file=sys.stderr)
        return 1
    finally:
        _close_store(store)


def _eval_run(args: argparse.Namespace) -> int:
    store = _choose_store(args.store, args.store_type)
    try:
        payload = run_eval_suite(
            store,
            suite=args.suite,
            runs=args.runs,
            output_dir=args.output_dir,
        )
        if args.out:
            report_path = write_markdown_report(payload, args.out)
            payload["report"] = str(report_path)
        if args.json:
            _print_json(payload)
        else:
            aggregate = payload["aggregate"]
            print(f"eval suite: {payload['suite']}")
            print(f"case_runs: {aggregate['case_runs']}")
            print(f"success_rate: {aggregate['success_rate']}")
            print(f"task_completion_rate: {aggregate['task_completion_rate']}")
            print(f"recovery_success_rate: {aggregate['recovery_success_rate']}")
            print(f"duplicate_tool_executions: {aggregate['duplicate_tool_executions']}")
            print(f"total_tokens: {aggregate['total_tokens']}")
            print(f"state_overhead_ms_p50: {aggregate['state_overhead_ms_p50']}")
            if args.out:
                print(f"report: {payload['report']}")
        return 0 if payload["aggregate"]["failed"] == 0 else 1
    except Exception as exc:
        print(f"jade eval run failed: {exc}", file=sys.stderr)
        return 1
    finally:
        _close_store(store)


def _eval_report(args: argparse.Namespace) -> int:
    store = _choose_store(args.store, args.store_type)
    try:
        payload = build_eval_report_payload(store, suite=args.suite)
        if args.out:
            report_path = write_markdown_report(payload, args.out)
            payload["report"] = str(report_path)
        if args.json:
            _print_json(payload)
        else:
            aggregate = payload["aggregate"]
            print(f"eval report suite: {payload['suite']}")
            print(f"case_runs: {aggregate['case_runs']}")
            print(f"success_rate: {aggregate['success_rate']}")
            print(f"task_completion_rate: {aggregate['task_completion_rate']}")
            print(f"failed: {aggregate['failed']}")
            print(f"total_tokens: {aggregate['total_tokens']}")
            print(f"state_overhead_ms_p50: {aggregate['state_overhead_ms_p50']}")
            if args.out:
                print(f"report: {payload['report']}")
        return 0 if payload["aggregate"]["failed"] == 0 else 1
    except Exception as exc:
        print(f"jade eval report failed: {exc}", file=sys.stderr)
        return 1
    finally:
        _close_store(store)


def _load_example_module(relative_path: str) -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    path = root / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load example module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _demo_crash_recovery(args: argparse.Namespace) -> int:
    try:
        module = _load_example_module("examples/jgx_crash_recovery.py")
        result = module.run_demo()
        payload = {
            "run_id": result["run_id"],
            "state_path": result["state_path"],
            "output_path": result["output_path"],
            "latest_phase": result["inspect"]["latest_phase"],
            "snapshots": result["inspect"]["snapshot_count"],
            "events": result["inspect"]["event_count"],
        }
        if args.json:
            _print_json(payload)
        else:
            print("crash recovery demo completed")
            for key, value in payload.items():
                print(f"{key}: {value}")
        return 0
    except Exception as exc:
        print(f"jade demo crash-recovery failed: {exc}", file=sys.stderr)
        return 1


def _demo_mesh_code_project(args: argparse.Namespace) -> int:
    try:
        module = _load_example_module("examples/jgx_mesh_code_project.py")
        result = module.build_workflow()
        payload = {
            "workflow_id": result["workflow_id"],
            "project_root": result["project_root"],
            "model": result["model"],
            "mesh_capsules": len(result["mesh_capsules"]),
            "agent_capsules": len(result["agent_capsules"]),
            "test_returncode": result["test_returncode"],
        }
        if args.json:
            _print_json(payload)
        else:
            print("mesh code project demo completed")
            for key, value in payload.items():
                print(f"{key}: {value}")
        return 0
    except Exception as exc:
        print(f"jade demo mesh-code-project failed: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jade", description="JadeAgent command line tools")
    sub = parser.add_subparsers(dest="command")

    state = sub.add_parser("state", help="Inspect governed execution state")
    state_sub = state.add_subparsers(dest="state_command")

    def add_store_options(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--store",
            default=".jade_state",
            help="State store path. Directories use FileStateStore; .db/.sqlite files use SqliteStateStore.",
        )
        p.add_argument(
            "--store-type",
            choices=("auto", "file", "sqlite"),
            default="auto",
            help="Override automatic store type detection.",
        )
        p.add_argument("--json", action="store_true", help="Emit JSON")

    def add_run_store_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("run_id", help="JGX run id")
        add_store_options(p)

    inspect = state_sub.add_parser("inspect", help="Inspect a JGX run")
    add_run_store_args(inspect)
    inspect.set_defaults(func=_state_inspect)

    history = state_sub.add_parser("history", help="Show JGX event history")
    add_run_store_args(history)
    history.add_argument("--limit", type=int, default=20, help="Maximum events to show")
    history.set_defaults(func=_state_history)

    state_list = state_sub.add_parser("list", help="List JGX runs in a store")
    add_store_options(state_list)
    state_list.set_defaults(func=_state_list)

    latest = state_sub.add_parser("latest", help="Show the latest snapshot for a JGX run")
    add_run_store_args(latest)
    latest.set_defaults(func=_state_latest)

    export = state_sub.add_parser("export", help="Export a JGX run to a directory capsule")
    add_run_store_args(export)
    export.add_argument("--out", required=True, help="Output .jgx directory")
    export.set_defaults(func=_state_export)

    timeline = state_sub.add_parser("timeline", help="Show a merged event/snapshot timeline")
    add_run_store_args(timeline)
    timeline.add_argument("--html", help="Write an HTML timeline")
    timeline.set_defaults(func=_state_timeline)

    verify = state_sub.add_parser("verify", help="Verify JGX integrity and secret hygiene")
    add_run_store_args(verify)
    verify.set_defaults(func=_state_verify)

    demo = sub.add_parser("demo", help="Run JadeAgent demos")
    demo_sub = demo.add_subparsers(dest="demo_command")

    crash = demo_sub.add_parser("crash-recovery", help="Run SQLite JGX crash recovery demo")
    crash.add_argument("--json", action="store_true", help="Emit JSON")
    crash.set_defaults(func=_demo_crash_recovery)

    mesh_code = demo_sub.add_parser("mesh-code-project", help="Run medium mesh code project demo")
    mesh_code.add_argument("--json", action="store_true", help="Emit JSON")
    mesh_code.set_defaults(func=_demo_mesh_code_project)

    eval_parser = sub.add_parser("eval", help="Run and report JadeAgent eval suites")
    eval_sub = eval_parser.add_subparsers(dest="eval_command")

    eval_run = eval_sub.add_parser("run", help="Run a deterministic eval suite")
    eval_run.add_argument("--suite", choices=("fast", "reliability", "core"), default="core")
    eval_run.add_argument("--runs", type=int, default=1, help="Repetitions per case")
    eval_run.add_argument(
        "--output-dir",
        default="examples/generated/eval",
        help="Directory for eval artifacts",
    )
    eval_run.add_argument(
        "--out",
        default="docs/jgx-eval-report.md",
        help="Markdown report output path",
    )
    eval_run.add_argument(
        "--store",
        default="examples/generated/eval/state.sqlite3",
        help="SQLite state store path for eval runs",
    )
    eval_run.add_argument(
        "--store-type",
        choices=("auto", "file", "sqlite"),
        default="auto",
        help="Override automatic store type detection.",
    )
    eval_run.add_argument("--json", action="store_true", help="Emit JSON")
    eval_run.set_defaults(func=_eval_run)

    eval_report = eval_sub.add_parser("report", help="Build a report from stored eval runs")
    eval_report.add_argument("--suite", choices=("fast", "reliability", "core"), default=None)
    eval_report.add_argument(
        "--store",
        default="examples/generated/eval/state.sqlite3",
        help="SQLite state store path for eval runs",
    )
    eval_report.add_argument(
        "--store-type",
        choices=("auto", "file", "sqlite"),
        default="auto",
        help="Override automatic store type detection.",
    )
    eval_report.add_argument(
        "--out",
        default="docs/jgx-eval-report.md",
        help="Markdown report output path",
    )
    eval_report.add_argument("--json", action="store_true", help="Emit JSON")
    eval_report.set_defaults(func=_eval_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
