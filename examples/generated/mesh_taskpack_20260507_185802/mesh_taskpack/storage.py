"""Persistence helpers for mesh_taskpack."""

from __future__ import annotations

import json
from pathlib import Path

from .models import Status
from .planner import TaskPlanner


def save_project(planner: TaskPlanner, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(planner.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return output


def load_project(path: str | Path) -> TaskPlanner:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TaskPlanner.from_dict(data)


def export_markdown(planner: TaskPlanner) -> str:
    lines = [f"# {planner.name}", ""]
    for status in Status:
        tasks = planner.list_by_status(status)
        lines.append(f"## {status.value}")
        if not tasks:
            lines.append("- none")
        for task in tasks:
            deps = f" deps={','.join(task.depends_on)}" if task.depends_on else ""
            lines.append(f"- [{task.id}] {task.title} owner={task.owner} priority={int(task.priority)}{deps}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
