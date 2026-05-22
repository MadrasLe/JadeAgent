"""
Safety policy for simulated swarm operations.

The policy enforces hard constraints independently from LLM decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .drone import DroneState, DroneStatus


@dataclass
class SafetyLimits:
    """Hard operational constraints."""

    max_altitude: float = 60.0
    min_altitude: float = 0.0
    min_battery_takeoff: float = 20.0
    min_battery_move: float = 15.0
    min_battery_scan: float = 10.0
    min_battery_rth: float = 8.0
    geofence_x: tuple[float, float] = (-200.0, 200.0)
    geofence_y: tuple[float, float] = (-200.0, 200.0)


class SafetyPolicy:
    """Evaluates whether a command is allowed for a given drone."""

    def __init__(self, limits: SafetyLimits | None = None):
        self.limits = limits or SafetyLimits()
        self.kill_switch = False

    def enable_kill_switch(self):
        self.kill_switch = True

    def disable_kill_switch(self):
        self.kill_switch = False

    def validate(self, drone: DroneState, action: str, params: dict[str, Any]) -> tuple[bool, str]:
        if self.kill_switch:
            return False, "Kill switch enabled. Command rejected."

        if drone.status == DroneStatus.OFFLINE:
            return False, "Drone is offline."

        action = action.strip().lower()
        if action == "takeoff":
            altitude = float(params.get("altitude", 10.0))
            if drone.status != DroneStatus.IDLE:
                return False, "Takeoff allowed only when drone is idle."
            if drone.battery < self.limits.min_battery_takeoff:
                return False, "Battery below takeoff threshold."
            if not self._is_altitude_safe(altitude):
                return False, "Requested altitude violates safety limits."
            return True, "ok"

        if action == "goto":
            if drone.status == DroneStatus.IDLE:
                return False, "Drone must take off before moving."
            if drone.battery < self.limits.min_battery_move:
                return False, "Battery below movement threshold."
            x = float(params.get("x", drone.position.x))
            y = float(params.get("y", drone.position.y))
            z = float(params.get("z", drone.position.z))
            if not self._is_inside_geofence(x, y):
                return False, "Target is outside geofence."
            if not self._is_altitude_safe(z):
                return False, "Target altitude violates safety limits."
            return True, "ok"

        if action == "scan":
            if drone.status == DroneStatus.IDLE:
                return False, "Drone must be airborne to scan."
            if drone.battery < self.limits.min_battery_scan:
                return False, "Battery below scan threshold."
            return True, "ok"

        if action in ("land", "return_home"):
            if action == "return_home" and drone.battery < self.limits.min_battery_rth:
                return False, "Battery too low for return_home."
            return True, "ok"

        if action == "set_battery":
            return True, "ok"

        return False, f"Unknown or forbidden action '{action}'."

    def _is_inside_geofence(self, x: float, y: float) -> bool:
        return (
            self.limits.geofence_x[0] <= x <= self.limits.geofence_x[1]
            and self.limits.geofence_y[0] <= y <= self.limits.geofence_y[1]
        )

    def _is_altitude_safe(self, z: float) -> bool:
        return self.limits.min_altitude <= z <= self.limits.max_altitude

