"""
LLM tool adapters for controlling the swarm simulator.
"""

from __future__ import annotations

import json

from ..core.tools import Tool
from .simulator import SwarmSimulator


class SwarmToolset:
    """Tool wrapper exposing simulator commands with structured outputs."""

    def __init__(self, simulator: SwarmSimulator):
        self.sim = simulator

    def list_drones(self) -> str:
        """List all drones in the fleet."""
        return json.dumps({"drones": self.sim.list_drones()})

    def fleet_telemetry(self) -> str:
        """Return fleet-level telemetry summary."""
        return json.dumps({"telemetry": self.sim.telemetry()})

    def get_drone_status(self, drone_id: str) -> str:
        """Get status for one drone.

        Args:
            drone_id: Target drone identifier.
        """
        status = self.sim.get_drone(drone_id)
        if status is None:
            return json.dumps({"success": False, "error": f"Drone '{drone_id}' not found."})
        return json.dumps({"success": True, "drone": status})

    def takeoff(self, drone_id: str, altitude: float = 10.0) -> str:
        """Take off a drone to a target altitude.

        Args:
            drone_id: Target drone identifier.
            altitude: Target altitude in meters.
        """
        result = self.sim.command(drone_id, "takeoff", altitude=altitude)
        return json.dumps(result.as_dict())

    def goto(self, drone_id: str, x: float, y: float, z: float) -> str:
        """Move a drone to a target coordinate.

        Args:
            drone_id: Target drone identifier.
            x: Target X coordinate in meters.
            y: Target Y coordinate in meters.
            z: Target altitude in meters.
        """
        result = self.sim.command(drone_id, "goto", x=x, y=y, z=z)
        return json.dumps(result.as_dict())

    def scan(self, drone_id: str, target: str = "area") -> str:
        """Run a scan action with a drone.

        Args:
            drone_id: Target drone identifier.
            target: Label of the target area/object.
        """
        result = self.sim.command(drone_id, "scan", target=target)
        return json.dumps(result.as_dict())

    def land(self, drone_id: str) -> str:
        """Land a drone.

        Args:
            drone_id: Target drone identifier.
        """
        result = self.sim.command(drone_id, "land")
        return json.dumps(result.as_dict())

    def return_home(self, drone_id: str) -> str:
        """Command a drone to return to its home position.

        Args:
            drone_id: Target drone identifier.
        """
        result = self.sim.command(drone_id, "return_home")
        return json.dumps(result.as_dict())

    def set_drone_battery(self, drone_id: str, value: float) -> str:
        """Set drone battery for test scenarios.

        Args:
            drone_id: Target drone identifier.
            value: New battery level percentage (0 to 100).
        """
        result = self.sim.set_battery(drone_id, value)
        return json.dumps(result.as_dict())

    def set_kill_switch(self, enabled: bool) -> str:
        """Enable or disable global kill switch.

        Args:
            enabled: True to block all flight commands, False to unblock.
        """
        if enabled:
            self.sim.safety.enable_kill_switch()
        else:
            self.sim.safety.disable_kill_switch()
        return json.dumps({
            "success": True,
            "kill_switch": self.sim.safety.kill_switch,
        })

    def tick(self, seconds: float = 1.0) -> str:
        """Advance simulation time.

        Args:
            seconds: Number of seconds to advance.
        """
        self.sim.tick(seconds=seconds)
        return json.dumps({"success": True, "telemetry": self.sim.telemetry()})


def create_swarm_tools(simulator: SwarmSimulator, safe_mode: bool = False) -> list[Tool]:
    """
    Build Tool objects for the simulator.

    Args:
        simulator: SwarmSimulator instance.
        safe_mode: If True, asks confirmation before each tool execution.
    """
    toolset = SwarmToolset(simulator)
    method_names = [
        "list_drones",
        "fleet_telemetry",
        "get_drone_status",
        "takeoff",
        "goto",
        "scan",
        "land",
        "return_home",
        "set_drone_battery",
        "set_kill_switch",
        "tick",
    ]
    tools: list[Tool] = []
    for name in method_names:
        method = getattr(toolset, name)
        tools.append(Tool(
            method,
            name=name,
            safe_mode=safe_mode,
            resource_refs=[f"tool.execute:{name}"],
        ))
    return tools
