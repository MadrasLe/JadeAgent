"""
Run a Redis-backed mesh node where the node itself is an LLM agent.

Usage:
    set REDIS_URL=redis://127.0.0.1:6379/0
    set MESH_HMAC_SECRET=my_secret
    set MESH_NODE_ID=planner_1
    set MESH_CAPABILITIES=plan
    set OPENROUTER_API_KEY=sk-...
    set OPENROUTER_MODEL=openai/gpt-4o-mini
    python examples/redis_agent_node.py

Optional delegation:
    set MESH_DELEGATE_CAPABILITIES=research,code,sandbox_exec
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jadeagent import Agent
from jadeagent.backends import OpenAICompatBackend
from jadeagent.mesh import (
    DistributedMeshRouter,
    HMACSigner,
    MeshDelegationClient,
    MeshNode,
    RedisMeshTransport,
    ReplayConfig,
    ReplayProtector,
    make_agent_task_handler,
    make_mesh_delegate_tool,
)


def _parse_csv(raw: str | None, default: str = "") -> list[str]:
    tokens = [token.strip() for token in (raw or default).split(",")]
    return [token for token in tokens if token]


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def _build_backend() -> OpenAICompatBackend:
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY for redis_agent_node.")

    return OpenAICompatBackend(
        model=os.environ.get("OPENROUTER_MODEL", os.environ.get("OPENAI_MODEL", "openai/gpt-4o-mini")),
        base_url=os.environ.get("OPENROUTER_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")),
        api_key=api_key,
    )


def _build_system_prompt(node_id: str, capabilities: list[str], delegate_caps: list[str]) -> str:
    custom = os.environ.get("AGENT_SYSTEM_PROMPT")
    if custom:
        return custom

    prompt = (
        f"You are agent node {node_id} in a distributed mesh. "
        f"Your local capabilities are: {', '.join(capabilities)}. "
        "Be direct, practical, and concise."
    )
    if delegate_caps:
        prompt += (
            f" You can delegate work to other mesh agents for: {', '.join(delegate_caps)}. "
            "Use delegation tools when another node is more appropriate."
        )
    return prompt


def main():
    redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
    secret = os.environ.get("MESH_HMAC_SECRET", "")
    node_id = os.environ.get("MESH_NODE_ID", "agent_node_default")
    capabilities = _parse_csv(os.environ.get("MESH_CAPABILITIES"), default="general")
    delegate_caps = _parse_csv(os.environ.get("MESH_DELEGATE_CAPABILITIES"))
    delegate_timeout = float(os.environ.get("MESH_DELEGATE_TIMEOUT", "45"))

    if not secret:
        raise RuntimeError("Set MESH_HMAC_SECRET with a shared key.")
    if not capabilities:
        capabilities = ["general"]

    backend = _build_backend()

    signer = HMACSigner(secret=secret, key_id="prod_key")
    replay = ReplayProtector(ReplayConfig(max_age_seconds=180.0, max_skew_seconds=20.0))

    transport = RedisMeshTransport(
        redis_url=redis_url,
        channel_prefix="jade:secure:mesh:",
        tls=redis_url.startswith("rediss://"),
        signer=signer,
        replay_protector=replay,
        poll_timeout=0.1,
    )
    router = DistributedMeshRouter(
        redis_url=redis_url,
        registry_prefix="jade:secure:mesh:registry",
        stale_after=60.0,
        heartbeat_ttl=180,
        refresh_interval=0.2,
        tls=redis_url.startswith("rediss://"),
    )

    delegation_client = None
    tools = []
    if delegate_caps:
        delegation_client = MeshDelegationClient(
            router=router,
            bus=transport,
            node_id=f"{node_id}_delegate",
            capability="delegate",
            wait_route_seconds=float(os.environ.get("MESH_WAIT_ROUTE_SECONDS", "10")),
            route_poll_interval=float(os.environ.get("MESH_ROUTE_POLL_INTERVAL", "0.2")),
            default_task_ttl=int(os.environ.get("MESH_TASK_TTL", "8")),
        )
        descriptions_raw = os.environ.get("MESH_DELEGATE_DESCRIPTIONS_JSON", "")
        try:
            descriptions = json.loads(descriptions_raw) if descriptions_raw else {}
        except Exception:
            descriptions = {}
        if not isinstance(descriptions, dict):
            descriptions = {}

        for cap in delegate_caps:
            tools.append(
                make_mesh_delegate_tool(
                    submit_task=lambda capability, prompt, metadata, _timeout=delegate_timeout: delegation_client.submit_text(
                        capability=capability,
                        prompt=prompt,
                        metadata=metadata,
                        timeout_seconds=_timeout,
                    ),
                    capability=cap,
                    name=f"ask_{_slug(cap)}",
                    description=str(descriptions.get(cap) or f"Delegate work to capability '{cap}'."),
                )
            )

    agent = Agent(
        backend=backend,
        name=os.environ.get("AGENT_NAME", node_id),
        system_prompt=_build_system_prompt(node_id, capabilities, delegate_caps),
        tools=tools,
        max_iterations=int(os.environ.get("AGENT_MAX_ITERATIONS", "6")),
        temperature=float(os.environ.get("AGENT_TEMPERATURE", "0.2")),
        max_tokens=int(os.environ.get("AGENT_MAX_TOKENS", "1200")),
        verbose=os.environ.get("AGENT_VERBOSE", "false").strip().lower() == "true",
    )

    node = MeshNode(
        node_id=node_id,
        capabilities=set(capabilities),
        router=router,
        bus=transport,
        task_handler=make_agent_task_handler(agent, node_id),
        verbose=False,
    )

    print(
        f"[agent-node:{node_id}] online caps={capabilities} "
        f"delegate_caps={delegate_caps}"
    )
    try:
        while True:
            node.step()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print(f"\n[agent-node:{node_id}] shutting down")
    finally:
        if delegation_client is not None:
            delegation_client.close()
        try:
            router.unregister_node(node_id)
        except Exception:
            pass
        transport.close()


if __name__ == "__main__":
    main()
