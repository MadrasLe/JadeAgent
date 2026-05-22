"""
Mission controller demo with MCP-style data sources and mesh swarm execution.

Run:
    python examples/mission_mcp_sim.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jadeagent.mesh import InMemoryMeshBus, MeshNode, MeshRouter
from jadeagent.swarm import (
    InMemoryMissionDatabase,
    MissionController,
    SafetyLimits,
    SafetyPolicy,
    SimulatedWeatherFeed,
    SimulatorGPSFeed,
    SwarmMeshController,
    SwarmSimulator,
)


def build_swarm() -> SwarmSimulator:
    limits = SafetyLimits(
        max_altitude=55.0,
        min_battery_takeoff=22.0,
        geofence_x=(-150.0, 150.0),
        geofence_y=(-150.0, 150.0),
    )
    sim = SwarmSimulator(
        safety_policy=SafetyPolicy(limits=limits),
        comms_failure_rate=0.0,
        random_seed=3,
    )
    sim.add_drone("drone_alpha", home_x=-20, home_y=5, battery=95)
    sim.add_drone("drone_bravo", home_x=15, home_y=-8, battery=90)
    return sim


def main():
    simulator = build_swarm()
    swarm_controller = SwarmMeshController(simulator)

    router = MeshRouter(stale_after=180.0)
    bus = InMemoryMeshBus()

    coordinator = MeshNode(
        node_id="mission_coord",
        capabilities={"orchestrate"},
        router=router,
        bus=bus,
        task_handler=lambda task: f"coord ack {task.task_id}",
    )
    MeshNode(
        node_id="swarm_worker_1",
        capabilities={"swarm_command"},
        router=router,
        bus=bus,
        task_handler=swarm_controller.handle_task,
    )
    MeshNode(
        node_id="swarm_worker_2",
        capabilities={"swarm_command"},
        router=router,
        bus=bus,
        task_handler=swarm_controller.handle_task,
    )

    mission_db = InMemoryMissionDatabase(
        missions={
            "bridge_inspection_01": {
                "mission_id": "bridge_inspection_01",
                "drone_id": "drone_alpha",
                "target": {"x": 48, "y": 20, "z": 22},
                "scan_target": "bridge_support_pillar_a",
                "priority": 2,
            }
        }
    )
    weather = SimulatedWeatherFeed(random_seed=11, base_wind=5.5)
    gps = SimulatorGPSFeed(simulator)

    mission = MissionController(
        coordinator=coordinator,
        bus=bus,
        mission_db=mission_db,
        weather_feed=weather,
        gps_feed=gps,
        execution_capability="swarm_command",
        max_safe_wind_mps=16.0,
        max_safe_rain=0.7,
        verbose=True,
    )

    run = mission.run_mission("bridge_inspection_01")
    print("\n=== Mission Summary ===")
    print(json.dumps(run.to_dict(), indent=2))

    print("\n=== Mission Events (DB) ===")
    for row in mission_db.get_events("bridge_inspection_01"):
        print(json.dumps(row, indent=2))

    print("\n=== Fleet Telemetry ===")
    print(json.dumps(simulator.telemetry(), indent=2))

    print("\n=== Router Snapshot ===")
    for row in router.snapshot():
        print(
            f"  {row['node_id']}: caps={row['capabilities']} "
            f"load={row['load_factor']} inflight={row['inflight']} q={row['queue_depth']}"
        )


if __name__ == "__main__":
    main()

