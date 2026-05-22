"""Small command line interface for mesh_taskpack."""

from __future__ import annotations

import argparse

from .models import Priority, Status
from .planner import TaskPlanner
from .storage import export_markdown, load_project, save_project


def build_demo_project() -> TaskPlanner:
    planner = TaskPlanner(name="demo")
    planner.add_task("Write domain model", owner="domain", priority=Priority.HIGH)
    planner.add_task("Build planner", owner="engine", priority=Priority.HIGH, depends_on=("write-domain-model",))
    planner.add_task("Add tests", owner="qa", priority=Priority.CRITICAL, depends_on=("build-planner",))
    return planner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mesh-taskpack")
    parser.add_argument("--file", default="taskpack.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo")
    sub.add_parser("summary")
    add = sub.add_parser("add")
    add.add_argument("title")
    add.add_argument("--owner", default="unassigned")
    add.add_argument("--priority", type=int, default=int(Priority.MEDIUM))
    done = sub.add_parser("done")
    done.add_argument("task_id")
    args = parser.parse_args(argv)

    if args.command == "demo":
        planner = build_demo_project()
        save_project(planner, args.file)
        print(export_markdown(planner))
        return 0

    try:
        planner = load_project(args.file)
    except FileNotFoundError:
        planner = TaskPlanner()

    if args.command == "add":
        planner.add_task(args.title, owner=args.owner, priority=Priority(args.priority))
        save_project(planner, args.file)
        print(export_markdown(planner))
        return 0

    if args.command == "done":
        planner.complete_task(args.task_id)
        save_project(planner, args.file)
        print(export_markdown(planner))
        return 0

    if args.command == "summary":
        print(export_markdown(planner))
        risks = planner.risk_report()
        if any(risks.values()):
            print("Risks:", risks)
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
