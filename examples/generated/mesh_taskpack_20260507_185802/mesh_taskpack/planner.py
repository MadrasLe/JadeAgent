"""Planning engine for mesh_taskpack."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .models import Priority, Status, Task, TaskValidationError


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "task"


@dataclass
class TaskPlanner:
    name: str = "taskpack"
    tasks: dict[str, Task] = field(default_factory=dict)

    def add_task(
        self,
        title: str,
        *,
        id: str | None = None,
        description: str = "",
        owner: str = "unassigned",
        priority: Priority | int = Priority.MEDIUM,
        tags: Iterable[str] = (),
        depends_on: Iterable[str] = (),
    ) -> Task:
        task_id = id or self._next_id(title)
        missing = [dep for dep in depends_on if dep not in self.tasks]
        if missing:
            raise TaskValidationError(f"unknown dependencies: {', '.join(missing)}")
        task = Task(
            id=task_id,
            title=title,
            description=description,
            owner=owner,
            priority=Priority(priority),
            tags=tuple(tags),
            depends_on=tuple(depends_on),
        )
        if task.id in self.tasks:
            raise TaskValidationError(f"duplicate task id: {task.id}")
        self.tasks[task.id] = task
        return task

    def _next_id(self, title: str) -> str:
        base = _slugify(title)
        candidate = base
        index = 2
        while candidate in self.tasks:
            candidate = f"{base}-{index}"
            index += 1
        return candidate

    def get(self, task_id: str) -> Task:
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            raise TaskValidationError(f"unknown task id: {task_id}") from exc

    def set_status(self, task_id: str, status: Status | str) -> Task:
        task = self.get(task_id)
        if Status(status) == Status.DONE:
            missing = [dep for dep in task.depends_on if self.tasks[dep].status != Status.DONE]
            if missing:
                raise TaskValidationError(f"cannot complete with unfinished dependencies: {missing}")
        updated = task.with_status(status)
        self.tasks[task_id] = updated
        return updated

    def start_task(self, task_id: str) -> Task:
        return self.set_status(task_id, Status.DOING)

    def block_task(self, task_id: str) -> Task:
        return self.set_status(task_id, Status.BLOCKED)

    def complete_task(self, task_id: str) -> Task:
        return self.set_status(task_id, Status.DONE)

    def completed_ids(self) -> set[str]:
        return {task.id for task in self.tasks.values() if task.status == Status.DONE}

    def ready_tasks(self) -> list[Task]:
        completed = self.completed_ids()
        return sorted(
            [task for task in self.tasks.values() if task.is_ready(completed)],
            key=lambda task: (-int(task.priority), task.created_at),
        )

    def list_by_status(self, status: Status | str) -> list[Task]:
        wanted = Status(status)
        return [task for task in self.tasks.values() if task.status == wanted]

    def workload_by_owner(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for task in self.tasks.values():
            if task.status != Status.DONE:
                counts[task.owner] += 1
        return dict(sorted(counts.items()))

    def dependency_order(self) -> list[str]:
        visiting: set[str] = set()
        visited: set[str] = set()
        ordered: list[str] = []

        def visit(task_id: str) -> None:
            if task_id in visited:
                return
            if task_id in visiting:
                raise TaskValidationError(f"cycle detected at {task_id}")
            visiting.add(task_id)
            for dep in self.get(task_id).depends_on:
                visit(dep)
            visiting.remove(task_id)
            visited.add(task_id)
            ordered.append(task_id)

        for task_id in self.tasks:
            visit(task_id)
        return ordered

    def risk_report(self) -> dict[str, list[str]]:
        blocked = [task.id for task in self.tasks.values() if task.status == Status.BLOCKED]
        waiting = [
            task.id
            for task in self.tasks.values()
            if task.status == Status.TODO and task.depends_on and not task.is_ready(self.completed_ids())
        ]
        overloaded = [owner for owner, count in self.workload_by_owner().items() if count >= 3]
        return {
            "blocked": blocked,
            "waiting_on_dependencies": waiting,
            "overloaded_owners": overloaded,
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "tasks": [self.tasks[task_id].to_dict() for task_id in self.dependency_order()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskPlanner":
        planner = cls(name=str(data.get("name", "taskpack")))
        for raw_task in data.get("tasks", []):
            task = Task.from_dict(raw_task)
            planner.tasks[task.id] = task
        planner.dependency_order()
        return planner
