"""
Mission controller: plan and execute swarm tasks with mesh + data feeds.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..mesh import InMemoryMeshBus, MeshNode, MeshTask
from .mission_data import GPSFeed, MissionDatabase, WeatherFeed


@dataclass
class MissionStepResult:
    """Execution result for one mission command."""

    action: str
    drone_id: str
    success: bool
    message: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class MissionRun:
    """Final mission execution summary."""

    mission_id: str
    status: str
    steps: list[MissionStepResult] = field(default_factory=list)
    weather: dict[str, Any] = field(default_factory=dict)
    gps_start: dict[str, Any] = field(default_factory=dict)
    gps_end: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None

    def finalize(self, status: str, error: str | None = None):
        self.status = status
        self.error = error
        self.finished_at = time.time()

    @property
    def success(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "status": self.status,
            "steps": [
                {
                    "action": s.action,
                    "drone_id": s.drone_id,
                    "success": s.success,
                    "message": s.message,
                    "raw": s.raw,
                }
                for s in self.steps
            ],
            "weather": self.weather,
            "gps_start": self.gps_start,
            "gps_end": self.gps_end,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "success": self.success,
        }


class MissionController:
    """
    Orchestrates drone missions using live data feeds and mesh workers.

    Mission schema expected from MissionDatabase:
    {
      "mission_id": "bridge_01",
      "drone_id": "drone_01",
      "target": {"x": 40, "y": 22, "z": 20},
      "scan_target": "bridge_section_a",
      "priority": 1
    }
    """

    def __init__(
        self,
        coordinator: MeshNode,
        bus: InMemoryMeshBus,
        mission_db: MissionDatabase,
        weather_feed: WeatherFeed,
        gps_feed: GPSFeed,
        execution_capability: str = "swarm_command",
        max_safe_wind_mps: float = 14.0,
        max_safe_rain: float = 0.6,
        verbose: bool = True,
    ):
        self.coordinator = coordinator
        self.bus = bus
        self.db = mission_db
        self.weather = weather_feed
        self.gps = gps_feed
        self.execution_capability = execution_capability
        self.max_safe_wind_mps = max_safe_wind_mps
        self.max_safe_rain = max_safe_rain
        self.verbose = verbose

    def run_mission(self, mission_id: str) -> MissionRun:
        mission = self.db.get_mission(mission_id)
        run = MissionRun(mission_id=mission_id, status="running")

        if mission is None:
            run.finalize("failed", f"Mission '{mission_id}' not found.")
            self._log(mission_id, {"type": "mission_failed", "reason": run.error})
            return run

        drone_id = str(mission.get("drone_id", "")).strip()
        target = mission.get("target", {}) if isinstance(mission.get("target"), dict) else {}
        scan_target = str(mission.get("scan_target", "area"))
        target_x = float(target.get("x", 0.0))
        target_y = float(target.get("y", 0.0))
        target_z = float(target.get("z", 15.0))
        priority = int(mission.get("priority", 0))

        if not drone_id:
            run.finalize("failed", "Mission missing 'drone_id'.")
            self._log(mission_id, {"type": "mission_failed", "reason": run.error})
            return run

        run.gps_start = self.gps.get_position(drone_id) or {}
        run.weather = self.weather.get_weather(target_x, target_y)

        if not self._is_weather_safe(run.weather):
            run.finalize(
                "aborted_weather",
                f"Unsafe weather: wind={run.weather.get('wind_mps')} rain={run.weather.get('rain_intensity')}",
            )
            self._log(mission_id, {
                "type": "mission_aborted",
                "reason": run.error,
                "weather": run.weather,
            })
            return run

        self._log(mission_id, {
            "type": "mission_started",
            "drone_id": drone_id,
            "target": {"x": target_x, "y": target_y, "z": target_z},
            "scan_target": scan_target,
            "weather": run.weather,
        })

        commands = [
            {"drone_id": drone_id, "action": "takeoff", "altitude": target_z},
            {"drone_id": drone_id, "action": "goto", "x": target_x, "y": target_y, "z": target_z},
            {"drone_id": drone_id, "action": "scan", "target": scan_target},
            {"drone_id": drone_id, "action": "return_home"},
            {"drone_id": drone_id, "action": "land"},
        ]

        for command in commands:
            result = self._run_command(mission_id, command, priority=priority)
            run.steps.append(result)
            if not result.success:
                self._log(mission_id, {
                    "type": "command_failed",
                    "command": command,
                    "message": result.message,
                })
                self._try_emergency_recover(mission_id, drone_id, run)
                run.gps_end = self.gps.get_position(drone_id) or {}
                run.finalize("failed", f"Command '{command['action']}' failed: {result.message}")
                return run

        run.gps_end = self.gps.get_position(drone_id) or {}
        run.finalize("completed")
        self._log(mission_id, {"type": "mission_completed", "gps_end": run.gps_end})
        return run

    def _run_command(self, mission_id: str, command: dict[str, Any], priority: int = 0) -> MissionStepResult:
        drone_id = str(command.get("drone_id", ""))
        action = str(command.get("action", "unknown"))
        task = MeshTask(
            capability=self.execution_capability,
            prompt=json.dumps(command),
            priority=priority,
            affinity=f"drone:{drone_id}" if drone_id else None,
            metadata={"command": dict(command), "mission_id": mission_id},
        )
        task_id = self.coordinator.submit_task(task)
        self.bus.run_until_idle()
        result_obj = self.coordinator.get_result(task_id)

        if result_obj is None:
            return MissionStepResult(
                action=action,
                drone_id=drone_id,
                success=False,
                message="No result returned by mesh executor.",
            )

        payload = self._parse_payload(result_obj.output)
        success = bool(payload.get("success"))
        message = str(payload.get("message", payload.get("error", result_obj.error or "")))
        return MissionStepResult(
            action=action,
            drone_id=drone_id,
            success=success,
            message=message,
            raw=payload,
        )

    def _try_emergency_recover(self, mission_id: str, drone_id: str, run: MissionRun):
        for command in (
            {"drone_id": drone_id, "action": "return_home"},
            {"drone_id": drone_id, "action": "land"},
        ):
            result = self._run_command(mission_id, command, priority=9)
            run.steps.append(result)
            self._log(mission_id, {
                "type": "recovery_command",
                "command": command,
                "success": result.success,
                "message": result.message,
            })

    def _log(self, mission_id: str, event: dict[str, Any]):
        row = dict(event)
        row.setdefault("timestamp", time.time())
        self.db.save_mission_event(mission_id, row)
        if self.verbose:
            event_type = row.get("type", "event")
            print(f"[MissionController] {mission_id} {event_type}: {row}")

    def _is_weather_safe(self, weather: dict[str, Any]) -> bool:
        wind = float(weather.get("wind_mps", 999))
        rain = float(weather.get("rain_intensity", 1.0))
        return wind <= self.max_safe_wind_mps and rain <= self.max_safe_rain

    @staticmethod
    def _parse_payload(text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            return {"success": False, "message": f"Unexpected payload type: {type(parsed)}"}
        except Exception:
            return {"success": False, "message": text}

