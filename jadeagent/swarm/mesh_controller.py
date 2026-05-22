"""
Mesh task adapter for swarm simulator commands.
"""

from __future__ import annotations

import json
from typing import Any

from ..mesh.protocol import MeshTask
from .simulator import SwarmSimulator


class SwarmMeshController:
    """Convert mesh tasks into simulator commands."""

    def __init__(self, simulator: SwarmSimulator):
        self.sim = simulator

    def handle_task(self, task: MeshTask) -> str:
        """
        Execute a swarm command encoded in a mesh task.

        Expected shape:
            task.metadata["command"] = {
                "drone_id": "drone_01",
                "action": "takeoff",
                "altitude": 12
            }
        """
        cmd, error = self._extract_command(task)
        if error is not None:
            return json.dumps({
                "success": False,
                "task_id": task.task_id,
                "error": error,
            })

        drone_id = str(cmd.get("drone_id", ""))
        action = str(cmd.get("action", "")).strip().lower()
        if not drone_id or not action:
            return json.dumps({
                "success": False,
                "task_id": task.task_id,
                "error": "Command must include 'drone_id' and 'action'.",
            })

        params = {k: v for k, v in cmd.items() if k not in ("drone_id", "action")}
        result = self.sim.command(drone_id, action, **params)
        payload = result.as_dict()
        payload["task_id"] = task.task_id
        return json.dumps(payload)

    @staticmethod
    def _extract_command(task: MeshTask) -> tuple[dict[str, Any], str | None]:
        meta_cmd = task.metadata.get("command")
        if isinstance(meta_cmd, dict):
            return dict(meta_cmd), None

        prompt = task.prompt.strip()
        if not prompt:
            return {}, "Task has no prompt and no metadata.command."

        if prompt.startswith("{"):
            try:
                parsed = json.loads(prompt)
                if isinstance(parsed, dict):
                    return parsed, None
                return {}, "Prompt JSON must be an object."
            except json.JSONDecodeError as exc:
                return {}, f"Invalid JSON prompt: {exc}"

        # Fallback syntax:
        #   drone_01 takeoff altitude=12
        parts = prompt.split()
        if len(parts) < 2:
            return {}, "Prompt command format invalid. Expected '<drone_id> <action> ...'."

        drone_id, action = parts[0], parts[1]
        params: dict[str, Any] = {}
        for item in parts[2:]:
            if "=" not in item:
                continue
            key, raw = item.split("=", 1)
            params[key] = SwarmMeshController._coerce(raw)
        return {"drone_id": drone_id, "action": action, **params}, None

    @staticmethod
    def _coerce(value: str) -> Any:
        lower = value.lower()
        if lower in ("true", "false"):
            return lower == "true"
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

