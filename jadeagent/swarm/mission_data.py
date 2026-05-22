"""
Mission data sources for swarm orchestration.

This module provides:
- Protocols for mission DB, weather feed and GPS feed.
- In-memory mock implementations for local testing.
- Optional MCP adapters for production integrations.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..mcp.client import MCPClient
    from .simulator import SwarmSimulator


class MissionDatabase(Protocol):
    """Mission persistence interface."""

    def get_mission(self, mission_id: str) -> dict[str, Any] | None:
        ...

    def save_mission_event(self, mission_id: str, event: dict[str, Any]):
        ...

    def get_events(self, mission_id: str) -> list[dict[str, Any]]:
        ...


class WeatherFeed(Protocol):
    """Weather data provider interface."""

    def get_weather(self, x: float, y: float) -> dict[str, Any]:
        ...


class GPSFeed(Protocol):
    """GPS/position provider interface."""

    def get_position(self, drone_id: str) -> dict[str, Any] | None:
        ...


class InMemoryMissionDatabase:
    """Simple in-memory mission store for tests and demos."""

    def __init__(self, missions: dict[str, dict[str, Any]] | None = None):
        self._missions = dict(missions or {})
        self._events: dict[str, list[dict[str, Any]]] = {}

    def upsert_mission(self, mission_id: str, mission: dict[str, Any]):
        self._missions[mission_id] = dict(mission)

    def get_mission(self, mission_id: str) -> dict[str, Any] | None:
        mission = self._missions.get(mission_id)
        return dict(mission) if mission is not None else None

    def save_mission_event(self, mission_id: str, event: dict[str, Any]):
        row = dict(event)
        row.setdefault("timestamp", time.time())
        self._events.setdefault(mission_id, []).append(row)

    def get_events(self, mission_id: str) -> list[dict[str, Any]]:
        return list(self._events.get(mission_id, []))


class SimulatedWeatherFeed:
    """
    Lightweight deterministic weather model.

    Wind and rain vary by coordinates and internal RNG.
    """

    def __init__(self, random_seed: int = 17, base_wind: float = 6.0):
        self._rng = random.Random(random_seed)
        self.base_wind = float(base_wind)

    def get_weather(self, x: float, y: float) -> dict[str, Any]:
        x = float(x)
        y = float(y)
        wind = self.base_wind + abs(x) * 0.015 + abs(y) * 0.01 + self._rng.uniform(-1.0, 1.0)
        gust = wind + self._rng.uniform(0.5, 3.5)
        rain = max(0.0, min(1.0, 0.05 + abs(x - y) * 0.0008))
        return {
            "wind_mps": round(wind, 2),
            "gust_mps": round(gust, 2),
            "rain_intensity": round(rain, 3),
            "visibility": "good" if rain < 0.3 else "moderate",
        }


class SimulatorGPSFeed:
    """GPS provider backed by current simulator state."""

    def __init__(self, simulator: SwarmSimulator):
        self.sim = simulator

    def get_position(self, drone_id: str) -> dict[str, Any] | None:
        status = self.sim.get_drone(drone_id)
        if status is None:
            return None
        pos = status.get("position", {})
        return {
            "drone_id": drone_id,
            "x": pos.get("x", 0.0),
            "y": pos.get("y", 0.0),
            "z": pos.get("z", 0.0),
            "battery": status.get("battery", 0.0),
            "status": status.get("status", "unknown"),
        }


@dataclass
class MCPToolMap:
    """Tool names for MCP integrations."""

    mission_get: str = "mission_get"
    mission_log: str = "mission_log"
    weather_get: str = "weather_get"
    gps_get: str = "gps_get"


class MCPMissionDatabase:
    """Mission DB adapter over an MCP client."""

    def __init__(self, client: MCPClient, tools: MCPToolMap | None = None):
        self.client = client
        self.tools = tools or MCPToolMap()

    def get_mission(self, mission_id: str) -> dict[str, Any] | None:
        payload = _parse_json_result(
            self.client.call_tool(self.tools.mission_get, {"mission_id": mission_id})
        )
        if isinstance(payload, dict):
            if payload.get("found") is False:
                return None
            mission = payload.get("mission")
            if isinstance(mission, dict):
                return mission
            return payload
        return None

    def save_mission_event(self, mission_id: str, event: dict[str, Any]):
        self.client.call_tool(
            self.tools.mission_log,
            {"mission_id": mission_id, "event": event},
        )

    def get_events(self, mission_id: str) -> list[dict[str, Any]]:
        # Optional capability depending on MCP server; return empty by default.
        return []


class MCPWeatherFeed:
    """Weather adapter over MCP client."""

    def __init__(self, client: MCPClient, tools: MCPToolMap | None = None):
        self.client = client
        self.tools = tools or MCPToolMap()

    def get_weather(self, x: float, y: float) -> dict[str, Any]:
        payload = _parse_json_result(
            self.client.call_tool(self.tools.weather_get, {"x": x, "y": y})
        )
        return payload if isinstance(payload, dict) else {"wind_mps": 999.0, "error": str(payload)}


class MCPGPSFeed:
    """GPS adapter over MCP client."""

    def __init__(self, client: MCPClient, tools: MCPToolMap | None = None):
        self.client = client
        self.tools = tools or MCPToolMap()

    def get_position(self, drone_id: str) -> dict[str, Any] | None:
        payload = _parse_json_result(
            self.client.call_tool(self.tools.gps_get, {"drone_id": drone_id})
        )
        return payload if isinstance(payload, dict) else None


def _parse_json_result(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return value

