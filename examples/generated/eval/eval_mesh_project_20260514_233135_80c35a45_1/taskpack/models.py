from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    done: bool = False
    depends_on: tuple[str, ...] = field(default_factory=tuple)

    def complete(self, completed: set[str]) -> "Task":
        missing = [dep for dep in self.depends_on if dep not in completed]
        if missing:
            raise ValueError(f"missing dependencies: {missing}")
        return Task(self.id, self.title, True, self.depends_on)
