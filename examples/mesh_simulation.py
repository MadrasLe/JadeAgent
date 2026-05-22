"""
JadeAgent mesh simulation (5-node in-memory network).

Run:
    python examples/mesh_simulation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jadeagent.mesh import InMemoryMeshBus, MeshNode, MeshRouter, MeshTask


def make_handler(node_name: str):
    """Create a simple deterministic task handler for demo purposes."""

    def _handle(task: MeshTask) -> str:
        return (
            f"{node_name} handled capability='{task.capability}' "
            f"task_id={task.task_id} prompt='{task.prompt}'"
        )

    return _handle


def main():
    router = MeshRouter(stale_after=120.0)
    bus = InMemoryMeshBus()

    coordinator = MeshNode(
        node_id="node_coordinator",
        capabilities={"orchestrate"},
        router=router,
        bus=bus,
        task_handler=make_handler("node_coordinator"),
    )
    MeshNode(
        node_id="node_planner",
        capabilities={"plan"},
        router=router,
        bus=bus,
        task_handler=make_handler("node_planner"),
    )
    MeshNode(
        node_id="node_researcher",
        capabilities={"research"},
        router=router,
        bus=bus,
        task_handler=make_handler("node_researcher"),
    )
    MeshNode(
        node_id="node_coder",
        capabilities={"code"},
        router=router,
        bus=bus,
        task_handler=make_handler("node_coder"),
    )
    MeshNode(
        node_id="node_reviewer",
        capabilities={"review"},
        router=router,
        bus=bus,
        task_handler=make_handler("node_reviewer"),
    )

    tasks = [
        MeshTask(capability="plan", prompt="Design a rollout plan for mesh agents."),
        MeshTask(capability="research", prompt="Collect risks of decentralized task routing."),
        MeshTask(capability="code", prompt="Implement in-memory envelope dedup logic."),
        MeshTask(capability="review", prompt="Audit failure modes and propose mitigations."),
        MeshTask(capability="translate", prompt="This should fail because no node supports it."),
    ]

    task_ids = [coordinator.submit_task(task) for task in tasks]
    cycles = bus.run_until_idle()

    print(f"Simulation cycles: {cycles}")
    print("")
    for task_id in task_ids:
        result = coordinator.get_result(task_id)
        if result is None:
            print(f"{task_id}: no result collected")
            continue
        if result.success:
            print(f"{task_id}: OK via {result.node_id}")
            print(f"  output: {result.output}")
        else:
            print(f"{task_id}: FAIL via {result.node_id}")
            print(f"  error: {result.error}")

    print("")
    print("Router snapshot:")
    for row in router.snapshot():
        print(
            f"  {row['node_id']}: caps={row['capabilities']} "
            f"load={row['load_factor']} inflight={row['inflight']} q={row['queue_depth']}"
        )
    print("")
    print("Coordinator metrics:", coordinator.metrics)


if __name__ == "__main__":
    main()
