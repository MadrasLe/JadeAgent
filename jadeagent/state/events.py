"""Append-only execution events for Jade .jgx capsules."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class JadeStateEvent:
    """A single state-machine transition or runtime activity record."""

    event_type: str
    run_id: str = ""
    phase: str = ""
    step: int = 0
    message: str = ""
    actor: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_event_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "parent_event_id": self.parent_event_id,
            "run_id": self.run_id,
            "event_type": self.event_type,
            "phase": self.phase,
            "step": self.step,
            "message": self.message,
            "actor": self.actor,
            "timestamp": self.timestamp,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JadeStateEvent":
        return cls(
            event_id=str(data.get("event_id") or uuid.uuid4().hex),
            parent_event_id=str(data.get("parent_event_id", "")),
            run_id=str(data.get("run_id", "")),
            event_type=str(data.get("event_type", "")),
            phase=str(data.get("phase", "")),
            step=int(data.get("step", 0)),
            message=str(data.get("message", "")),
            actor=str(data.get("actor", "")),
            timestamp=float(data.get("timestamp", time.time())),
            payload=dict(data.get("payload", {})),
        )
