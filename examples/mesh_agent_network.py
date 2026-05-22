"""
Example: mesh network where each node is an agent.

Roles:
- planner: breaks down work
- researcher: gathers risks/facts
- coder: drafts implementation steps
- coordinator: delegates to the others using mesh tools

This example uses in-memory transport, so it runs without Redis.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jadeagent import Agent
from jadeagent.backends import OpenAICompatBackend
from jadeagent.mesh import (
    InMemoryMeshBus,
    MeshNode,
    MeshRouter,
    MeshTask,
    make_agent_task_handler,
    make_mesh_delegate_tool,
)


def _submit_via_mesh(coordinator: MeshNode, bus: InMemoryMeshBus, capability: str, prompt: str) -> str:
    task = MeshTask(
        capability=capability,
        prompt=prompt,
        requester=coordinator.node_id,
        ttl=6,
    )
    task_id = coordinator.submit_task(task)
    bus.run_until_idle()
    result = coordinator.get_result(task_id)
    if result is None:
        return f"Delegation failed: no result for capability '{capability}'."
    if result.output:
        return result.output
    return result.error or f"Delegation failed for capability '{capability}'."


def _build_backend() -> OpenAICompatBackend:
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY for mesh_agent_network example.")
    return OpenAICompatBackend(
        model=model,
        base_url=base_url,
        api_key=api_key,
    )


def _extract_answer(raw_text: str) -> str:
    try:
        payload = json.loads(raw_text)
    except Exception:
        return raw_text
    if isinstance(payload, dict):
        return str(payload.get("answer") or payload.get("output") or raw_text)
    return raw_text


def main():
    backend = _build_backend()
    bus = InMemoryMeshBus()
    router = MeshRouter()

    planner_agent = Agent(
        backend=backend,
        name="planner_agent",
        system_prompt="You are a planning agent. Break requests into concrete steps.",
        verbose=False,
    )
    researcher_agent = Agent(
        backend=backend,
        name="researcher_agent",
        system_prompt="You are a research agent. Identify risks, unknowns, constraints, and facts.",
        verbose=False,
    )
    coder_agent = Agent(
        backend=backend,
        name="coder_agent",
        system_prompt="You are an implementation agent. Propose practical implementation details.",
        verbose=False,
    )

    planner_node = MeshNode(
        node_id="planner_node",
        capabilities={"plan"},
        router=router,
        bus=bus,
        task_handler=make_agent_task_handler(planner_agent, "planner_node"),
        verbose=False,
    )
    researcher_node = MeshNode(
        node_id="researcher_node",
        capabilities={"research"},
        router=router,
        bus=bus,
        task_handler=make_agent_task_handler(researcher_agent, "researcher_node"),
        verbose=False,
    )
    coder_node = MeshNode(
        node_id="coder_node",
        capabilities={"code"},
        router=router,
        bus=bus,
        task_handler=make_agent_task_handler(coder_agent, "coder_node"),
        verbose=False,
    )

    coordinator_stub = MeshNode(
        node_id="coordinator_node",
        capabilities={"orchestrate"},
        router=router,
        bus=bus,
        task_handler=lambda task: "coord stub",
        verbose=False,
    )

    delegate_plan = make_mesh_delegate_tool(
        submit_task=lambda cap, prompt, meta: _extract_answer(_submit_via_mesh(coordinator_stub, bus, cap, prompt)),
        capability="plan",
        name="ask_planner",
        description="Ask the planner agent to break down a task.",
    )
    delegate_research = make_mesh_delegate_tool(
        submit_task=lambda cap, prompt, meta: _extract_answer(_submit_via_mesh(coordinator_stub, bus, cap, prompt)),
        capability="research",
        name="ask_researcher",
        description="Ask the researcher agent for risks and constraints.",
    )
    delegate_code = make_mesh_delegate_tool(
        submit_task=lambda cap, prompt, meta: _extract_answer(_submit_via_mesh(coordinator_stub, bus, cap, prompt)),
        capability="code",
        name="ask_coder",
        description="Ask the coder agent for implementation details.",
    )

    coordinator_agent = Agent(
        backend=backend,
        name="coordinator_agent",
        system_prompt=(
            "You are a coordinator in a mesh of agents. "
            "Use the available tools to delegate planning, research, and code work. "
            "Then synthesize a final answer."
        ),
        tools=[delegate_plan, delegate_research, delegate_code],
        max_iterations=6,
        verbose=True,
    )
    coordinator_stub.task_handler = make_agent_task_handler(coordinator_agent, "coordinator_node")

    prompt = (
        "Design a useful multi-agent mesh system where each node is an agent, "
        "not just a worker. Include plan, risks, and implementation direction."
    )
    task = MeshTask(
        capability="orchestrate",
        prompt=prompt,
        requester="coordinator_node",
        ttl=6,
    )
    task_id = coordinator_stub.submit_task(task)

    started = time.time()
    while time.time() - started < 15.0:
        bus.run_until_idle()
        result = coordinator_stub.get_result(task_id)
        if result is not None:
            print("\n=== Final Result ===")
            print(result.output or result.error)
            break
        time.sleep(0.05)
    else:
        print("Timed out waiting for coordinator result.")

    print("\n=== Router Snapshot ===")
    for row in router.snapshot():
        print(
            f"  {row['node_id']}: caps={row['capabilities']} "
            f"load={row['load_factor']} inflight={row['inflight']} q={row['queue_depth']}"
        )


if __name__ == "__main__":
    main()
