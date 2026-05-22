"""
In-memory swarm simulator for multi-device orchestration tests.
"""

from __future__ import annotations

import random
import time
from typing import Any

from .drone import CommandResult, DroneState, DroneStatus, Position
from .safety import SafetyPolicy


class SwarmSimulator:
    """
    Simulate a drone fleet with deterministic command execution.

    This environment is designed for rapid integration tests with LLM tools and
    mesh orchestration, not flight-physics realism.
    """

    def __init__(
        self,
        drones: list[DroneState] | None = None,
        safety_policy: SafetyPolicy | None = None,
        comms_failure_rate: float = 0.0,
        random_seed: int = 7,
        max_events: int = 500,
    ):
        self.safety = safety_policy or SafetyPolicy()
        self.comms_failure_rate = max(0.0, min(1.0, float(comms_failure_rate)))
        self._rng = random.Random(random_seed)
        self._max_events = max_events
        self._drones: dict[str, DroneState] = {}
        self._events: list[dict[str, Any]] = []

        for drone in drones or []:
            self._drones[drone.drone_id] = drone

    def add_drone(
        self,
        drone_id: str,
        home_x: float = 0.0,
        home_y: float = 0.0,
        battery: float = 100.0,
    ):
        home = Position(home_x, home_y, 0.0)
        state = DroneState(
            drone_id=drone_id,
            home=home,
            position=Position(home_x, home_y, 0.0),
            battery=self._clamp_battery(battery),
        )
        self._drones[drone_id] = state

    def has_drone(self, drone_id: str) -> bool:
        return drone_id in self._drones

    def list_drones(self) -> list[dict[str, Any]]:
        return [d.snapshot() for d in sorted(self._drones.values(), key=lambda x: x.drone_id)]

    def get_drone(self, drone_id: str) -> dict[str, Any] | None:
        drone = self._drones.get(drone_id)
        return drone.snapshot() if drone else None

    def telemetry(self) -> dict[str, Any]:
        fleet_size = len(self._drones)
        airborne = sum(1 for d in self._drones.values() if d.status == DroneStatus.AIRBORNE)
        returning = sum(1 for d in self._drones.values() if d.status == DroneStatus.RETURNING)
        offline = sum(1 for d in self._drones.values() if d.status == DroneStatus.OFFLINE)
        avg_battery = (
            round(sum(d.battery for d in self._drones.values()) / fleet_size, 2)
            if fleet_size > 0
            else 0.0
        )
        return {
            "fleet_size": fleet_size,
            "airborne": airborne,
            "returning": returning,
            "offline": offline,
            "avg_battery": avg_battery,
            "events": len(self._events),
            "kill_switch": self.safety.kill_switch,
        }

    def recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return self._events[-limit:]

    def set_battery(self, drone_id: str, value: float) -> CommandResult:
        return self.command(drone_id, "set_battery", value=value)

    def command(self, drone_id: str, action: str, **params) -> CommandResult:
        drone = self._drones.get(drone_id)
        action = action.strip().lower()

        if drone is None:
            result = CommandResult(
                success=False,
                message=f"Drone '{drone_id}' not found.",
                drone_id=drone_id,
                action=action,
            )
            self._record_event(result, params)
            return result

        if self._rng.random() < self.comms_failure_rate:
            result = CommandResult(
                success=False,
                message=f"Simulated comms drop for action '{action}'.",
                drone_id=drone_id,
                action=action,
            )
            self._record_event(result, params)
            return result

        allowed, reason = self.safety.validate(drone, action, params)
        if not allowed:
            result = CommandResult(
                success=False,
                message=reason,
                drone_id=drone_id,
                action=action,
                data={"state": drone.snapshot()},
            )
            self._record_event(result, params)
            return result

        handler = getattr(self, f"_do_{action}", None)
        if handler is None:
            result = CommandResult(
                success=False,
                message=f"Unsupported action '{action}'.",
                drone_id=drone_id,
                action=action,
            )
            self._record_event(result, params)
            return result

        result = handler(drone, **params)
        drone.last_command_at = time.time()
        self._record_event(result, params)
        return result

    def tick(self, seconds: float = 1.0):
        """
        Advance simulation time with idle battery drain.
        """
        seconds = max(0.0, float(seconds))
        for drone in self._drones.values():
            if drone.status in (DroneStatus.AIRBORNE, DroneStatus.RETURNING):
                self._consume_battery(drone, 0.03 * seconds)

    def _do_takeoff(self, drone: DroneState, altitude: float = 10.0, **_) -> CommandResult:
        altitude = float(altitude)
        drone.position.z = altitude
        drone.status = DroneStatus.AIRBORNE
        self._consume_battery(drone, 1.5 + altitude * 0.02)
        return CommandResult(
            success=True,
            message=f"Takeoff complete at {altitude:.1f}m.",
            drone_id=drone.drone_id,
            action="takeoff",
            data={"state": drone.snapshot()},
        )

    def _do_goto(self, drone: DroneState, x: float, y: float, z: float | None = None, **_) -> CommandResult:
        target = Position(float(x), float(y), float(drone.position.z if z is None else z))
        distance = drone.position.distance_to(target)
        drone.position = target
        drone.status = DroneStatus.AIRBORNE
        self._consume_battery(drone, 0.06 * distance + 0.8)
        return CommandResult(
            success=True,
            message=f"Moved {distance:.1f}m to ({target.x:.1f}, {target.y:.1f}, {target.z:.1f}).",
            drone_id=drone.drone_id,
            action="goto",
            data={"distance": round(distance, 2), "state": drone.snapshot()},
        )

    def _do_scan(self, drone: DroneState, target: str = "area", **_) -> CommandResult:
        self._consume_battery(drone, 2.5)
        scan_id = f"scan_{int(time.time() * 1000)}"
        return CommandResult(
            success=True,
            message=f"Scan complete for target '{target}'.",
            drone_id=drone.drone_id,
            action="scan",
            data={"scan_id": scan_id, "target": target, "state": drone.snapshot()},
        )

    def _do_land(self, drone: DroneState, **_) -> CommandResult:
        drone.position.z = 0.0
        drone.status = DroneStatus.IDLE
        self._consume_battery(drone, 0.7)
        return CommandResult(
            success=True,
            message="Landing complete.",
            drone_id=drone.drone_id,
            action="land",
            data={"state": drone.snapshot()},
        )

    def _do_return_home(self, drone: DroneState, **_) -> CommandResult:
        drone.status = DroneStatus.RETURNING
        distance = drone.position.distance_to(drone.home)
        drone.position = Position(drone.home.x, drone.home.y, 0.0)
        drone.status = DroneStatus.IDLE
        self._consume_battery(drone, 0.07 * distance + 1.0)
        return CommandResult(
            success=True,
            message=f"Returned home after {distance:.1f}m.",
            drone_id=drone.drone_id,
            action="return_home",
            data={"distance": round(distance, 2), "state": drone.snapshot()},
        )

    def _do_set_battery(self, drone: DroneState, value: float, **_) -> CommandResult:
        drone.battery = self._clamp_battery(float(value))
        if drone.battery <= 0.0:
            drone.status = DroneStatus.OFFLINE
        elif drone.status == DroneStatus.OFFLINE:
            drone.status = DroneStatus.IDLE
        return CommandResult(
            success=True,
            message=f"Battery set to {drone.battery:.1f}%.",
            drone_id=drone.drone_id,
            action="set_battery",
            data={"state": drone.snapshot()},
        )

    def _consume_battery(self, drone: DroneState, amount: float):
        drone.battery = self._clamp_battery(drone.battery - float(amount))
        if drone.battery <= 0.0:
            drone.status = DroneStatus.OFFLINE

    def _record_event(self, result: CommandResult, params: dict[str, Any]):
        self._events.append({
            "timestamp": result.timestamp,
            "drone_id": result.drone_id,
            "action": result.action,
            "success": result.success,
            "message": result.message,
            "params": dict(params),
            "data": result.data,
        })
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

    @staticmethod
    def _clamp_battery(value: float) -> float:
        return max(0.0, min(100.0, float(value)))

