"""
Distributed swarm simulation with mesh routing and safety gates.

Run:
    python examples/swarm_sim.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jadeagent.mesh import InMemoryMeshBus, MeshNode, MeshRouter, MeshTask
from jadeagent.swarm import (
    SafetyLimits,
    SafetyPolicy,
    SwarmMeshController,
    SwarmSimulator,
    create_swarm_tools,
)


def build_simulator() -> SwarmSimulator:
    limits = SafetyLimits(
        max_altitude=50.0,
        min_battery_takeoff=25.0,
        min_battery_move=18.0,
        geofence_x=(-120.0, 120.0),
        geofence_y=(-120.0, 120.0),
    )
    safety = SafetyPolicy(limits=limits)
    sim = SwarmSimulator(safety_policy=safety, comms_failure_rate=0.0, random_seed=42)

    homes = [
        ("drone_01", -30, -10, 100),
        ("drone_02", -10, 0, 96),
        ("drone_03", 10, 5, 88),
        ("drone_04", 25, -5, 90),
        ("drone_05", 35, 15, 92),
        ("drone_06", 45, -20, 94),
    ]
    for drone_id, x, y, battery in homes:
        sim.add_drone(drone_id, home_x=x, home_y=y, battery=battery)
    return sim


def main():
    simulator = build_simulator()
    controller = SwarmMeshController(simulator)

    router = MeshRouter(stale_after=120.0)
    bus = InMemoryMeshBus()

    coordinator = MeshNode(
        node_id="coord",
        capabilities={"orchestrate"},
        router=router,
        bus=bus,
        task_handler=lambda task: f"orchestrator acknowledged {task.task_id}",
    )
    MeshNode(
        node_id="swarm_exec_a",
        capabilities={"swarm_command"},
        router=router,
        bus=bus,
        task_handler=controller.handle_task,
    )
    MeshNode(
        node_id="swarm_exec_b",
        capabilities={"swarm_command"},
        router=router,
        bus=bus,
        task_handler=controller.handle_task,
    )

    tasks = [
        MeshTask(
            capability="swarm_command",
            prompt="takeoff drone_01",
            metadata={"command": {"drone_id": "drone_01", "action": "takeoff", "altitude": 20}},
        ),
        MeshTask(
            capability="swarm_command",
            prompt="goto drone_01",
            metadata={"command": {"drone_id": "drone_01", "action": "goto", "x": 40, "y": 22, "z": 20}},
        ),
        MeshTask(
            capability="swarm_command",
            prompt="scan drone_01",
            metadata={"command": {"drone_id": "drone_01", "action": "scan", "target": "bridge_section_a"}},
        ),
        MeshTask(
            capability="swarm_command",
            prompt="takeoff drone_02",
            metadata={"command": {"drone_id": "drone_02", "action": "takeoff", "altitude": 25}},
        ),
        MeshTask(
            capability="swarm_command",
            prompt="unsafe geofence test",
            metadata={"command": {"drone_id": "drone_02", "action": "goto", "x": 300, "y": 0, "z": 20}},
        ),
        MeshTask(
            capability="swarm_command",
            prompt="force low battery for test",
            metadata={"command": {"drone_id": "drone_03", "action": "set_battery", "value": 12}},
        ),
        MeshTask(
            capability="swarm_command",
            prompt="battery safety test",
            metadata={"command": {"drone_id": "drone_03", "action": "takeoff", "altitude": 10}},
        ),
        MeshTask(
            capability="swarm_command",
            prompt="return and land",
            metadata={"command": {"drone_id": "drone_01", "action": "return_home"}},
        ),
        MeshTask(
            capability="swarm_command",
            prompt="land",
            metadata={"command": {"drone_id": "drone_01", "action": "land"}},
        ),
    ]

    task_ids = [coordinator.submit_task(task) for task in tasks]
    cycles = bus.run_until_idle()

    print(f"Cycles executed: {cycles}")
    print("")
    print("Task results:")
    for task_id in task_ids:
        result = coordinator.get_result(task_id)
        if result is None:
            print(f"  - {task_id}: missing result")
            continue

        try:
            payload = json.loads(result.output) if result.output else {}
        except json.JSONDecodeError:
            payload = {"success": False, "message": result.output}

        status = "OK" if payload.get("success") else "FAIL"
        message = payload.get("message", payload.get("error", "no message"))
        node_id = payload.get("drone_id", result.node_id)
        action = payload.get("action", "n/a")
        print(f"  - {task_id}: {status} action={action} target={node_id} msg={message}")

    print("")
    print("Fleet telemetry:")
    print(json.dumps(simulator.telemetry(), indent=2))

    print("")
    print("Sample drone states:")
    for drone_id in ("drone_01", "drone_02", "drone_03"):
        print(json.dumps(simulator.get_drone(drone_id), indent=2))

    print("")
    print("Toolset available for API-based LLM agents:")
    tools = create_swarm_tools(simulator)
    print([tool.name for tool in tools])

    print("")
    print("Router snapshot:")
    for row in router.snapshot():
        print(
            f"  {row['node_id']}: caps={row['capabilities']} "
            f"load={row['load_factor']} inflight={row['inflight']} q={row['queue_depth']}"
        )


if __name__ == "__main__":
    main()

