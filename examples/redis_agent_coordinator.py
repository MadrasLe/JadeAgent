"""
Run a local coordinator agent that delegates work to Redis-backed mesh agents.

Usage:
    set REDIS_URL=redis://127.0.0.1:6379/0
    set MESH_HMAC_SECRET=my_secret
    set OPENROUTER_API_KEY=sk-...
    set MESH_DELEGATE_CAPABILITIES=plan,research,code
    python examples/redis_agent_coordinator.py "Design a mesh architecture for autonomous agents."
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jadeagent import Agent
from jadeagent.backends import OpenAICompatBackend
from jadeagent.mesh import (
    DistributedMeshRouter,
    HMACSigner,
    MeshDelegationClient,
    RedisMeshTransport,
    ReplayConfig,
    ReplayProtector,
    make_mesh_delegate_tool,
)


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coordinator agent over Redis mesh.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Design a useful distributed agent mesh and propose next steps.",
        help="Task for the coordinator agent.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="Per-delegation timeout in seconds.",
    )
    parser.add_argument(
        "--delegate-capabilities",
        default=os.environ.get("MESH_DELEGATE_CAPABILITIES", "plan,research,code"),
        help="Comma-separated capabilities available in the mesh.",
    )
    return parser.parse_args()


def _build_backend() -> OpenAICompatBackend:
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY for redis_agent_coordinator.")

    return OpenAICompatBackend(
        model=os.environ.get("OPENROUTER_MODEL", os.environ.get("OPENAI_MODEL", "openai/gpt-4o-mini")),
        base_url=os.environ.get("OPENROUTER_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")),
        api_key=api_key,
    )


def main():
    args = _parse_args()
    redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
    secret = os.environ.get("MESH_HMAC_SECRET", "")
    if not secret:
        raise RuntimeError("Set MESH_HMAC_SECRET with a shared key.")

    delegate_caps = [token.strip() for token in args.delegate_capabilities.split(",") if token.strip()]
    if not delegate_caps:
        raise RuntimeError("Set at least one delegate capability.")

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

    delegation_client = MeshDelegationClient(
        router=router,
        bus=transport,
        node_id=f"coord_agent_{os.getpid()}",
        capability="orchestrate",
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

    tools = []
    for cap in delegate_caps:
        tools.append(
            make_mesh_delegate_tool(
                submit_task=lambda capability, prompt, metadata, _timeout=args.timeout: delegation_client.submit_text(
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
        name=os.environ.get("AGENT_NAME", "coordinator_agent"),
        system_prompt=os.environ.get(
            "AGENT_SYSTEM_PROMPT",
            (
                "You are a coordinator agent in a distributed mesh. "
                "Use delegation tools to call the right specialist nodes. "
                "Synthesize a final answer with plan, risks, and implementation details."
            ),
        ),
        tools=tools,
        max_iterations=int(os.environ.get("AGENT_MAX_ITERATIONS", "6")),
        temperature=float(os.environ.get("AGENT_TEMPERATURE", "0.2")),
        max_tokens=int(os.environ.get("AGENT_MAX_TOKENS", "1400")),
        verbose=True,
    )

    try:
        result = agent.run(args.prompt)
        print("\n=== Final Answer ===")
        print(result.answer)
        print("\n=== Tool Calls ===")
        for tc in result.tool_calls_made:
            print(f"- {tc.name} {tc.arguments}")
        print("\n=== Router Snapshot ===")
        for row in router.snapshot():
            print(
                f"  {row['node_id']}: caps={row['capabilities']} "
                f"load={row['load_factor']} inflight={row['inflight']} q={row['queue_depth']}"
            )
    finally:
        delegation_client.close()
        transport.close()


if __name__ == "__main__":
    main()
