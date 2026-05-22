"""Domain models for mesh_taskpack."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskValidationError(ValueError):
    """Raised when task data is invalid."""


class Status(str, Enum):
    TODO = "todo"
    DOING = "doing"
    BLOCKED = "blocked"
    DONE = "done"


class Priority(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    description: str = ""
    owner: str = "unassigned"
    priority: Priority = Priority.MEDIUM
    status: Status = Status.TODO
    tags: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise TaskValidationError("task id is required")
        if not self.title or not self.title.strip():
            raise TaskValidationError("task title is required")
        object.__setattr__(self, "id", self.id.strip())
        object.__setattr__(self, "title", self.title.strip())
        object.__setattr__(self, "owner", (self.owner or "unassigned").strip())
        object.__setattr__(self, "tags", tuple(str(tag).strip() for tag in self.tags if str(tag).strip()))
        object.__setattr__(
            self,
            "depends_on",
            tuple(str(dep).strip() for dep in self.depends_on if str(dep).strip()),
        )
        if isinstance(self.priority, int) and not isinstance(self.priority, Priority):
            object.__setattr__(self, "priority", Priority(self.priority))
        if isinstance(self.status, str):
            object.__setattr__(self, "status", Status(self.status))

    def with_status(self, status: Status | str) -> "Task":
        return Task(
            id=self.id,
            title=self.title,
            description=self.description,
            owner=self.owner,
            priority=self.priority,
            status=Status(status),
            tags=self.tags,
            depends_on=self.depends_on,
            created_at=self.created_at,
            updated_at=time.time(),
        )

    def is_ready(self, completed_ids: set[str]) -> bool:
        return self.status == Status.TODO and all(dep in completed_ids for dep in self.depends_on)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "owner": self.owner,
            "priority": int(self.priority),
            "status": self.status.value,
            "tags": list(self.tags),
            "depends_on": list(self.depends_on),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        return cls(
            id=str(data["id"]),
            title=str(data["title"]),
            description=str(data.get("description", "")),
            owner=str(data.get("owner", "unassigned")),
            priority=Priority(int(data.get("priority", Priority.MEDIUM))),
            status=Status(str(data.get("status", Status.TODO.value))),
            tags=tuple(data.get("tags", ())),
            depends_on=tuple(data.get("depends_on", ())),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
        )
