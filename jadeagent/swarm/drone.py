"""
Core drone state and command result models for swarm simulation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DroneStatus(str, Enum):
    """Runtime status for a simulated drone."""

    IDLE = "idle"
    AIRBORNE = "airborne"
    RETURNING = "returning"
    OFFLINE = "offline"


@dataclass
class Position:
    """3D position in meters."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def distance_to(self, other: Position) -> float:
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )

    def as_dict(self) -> dict[str, float]:
        return {"x": round(self.x, 2), "y": round(self.y, 2), "z": round(self.z, 2)}


@dataclass
class DroneState:
    """Complete mutable state for one simulated drone."""

    drone_id: str
    home: Position = field(default_factory=Position)
    position: Position = field(default_factory=Position)
    battery: float = 100.0
    status: DroneStatus = DroneStatus.IDLE
    last_command_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "drone_id": self.drone_id,
            "position": self.position.as_dict(),
            "home": self.home.as_dict(),
            "battery": round(self.battery, 2),
            "status": self.status.value,
            "last_command_at": self.last_command_at,
            "metadata": dict(self.metadata),
        }


@dataclass
class CommandResult:
    """Output returned by one simulator command."""

    success: bool
    message: str
    drone_id: str
    action: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "drone_id": self.drone_id,
            "action": self.action,
            "data": self.data,
            "timestamp": self.timestamp,
        }

