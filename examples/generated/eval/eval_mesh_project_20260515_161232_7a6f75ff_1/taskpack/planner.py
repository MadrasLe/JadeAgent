from __future__ import annotations

from .models import Task


class Planner:
    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}

    def add(self, task: Task) -> None:
        if task.id in self.tasks:
            raise ValueError(f"duplicate task: {task.id}")
        missing = [dep for dep in task.depends_on if dep not in self.tasks]
        if missing:
            raise ValueError(f"unknown dependencies: {missing}")
        self.tasks[task.id] = task

    def ready(self) -> list[str]:
        completed = {task.id for task in self.tasks.values() if task.done}
        return [
            task.id
            for task in self.tasks.values()
            if not task.done and all(dep in completed for dep in task.depends_on)
        ]

    def complete(self, task_id: str) -> None:
        completed = {task.id for task in self.tasks.values() if task.done}
        self.tasks[task_id] = self.tasks[task_id].complete(completed)

    def summary(self) -> dict[str, int]:
        done = sum(1 for task in self.tasks.values() if task.done)
        return {"total": len(self.tasks), "done": done, "open": len(self.tasks) - done}
