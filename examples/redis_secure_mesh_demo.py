"""
Secure Redis mesh transport demo.

Requirements:
    pip install -e ".[network]"
    # Redis server reachable at REDIS_URL

Environment variables:
    REDIS_URL: Redis connection string (default redis://localhost:6379/0)
    MESH_HMAC_SECRET: Shared HMAC key for all nodes (required)

Run:
    python examples/redis_secure_mesh_demo.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jadeagent.mesh import (
    DistributedMeshRouter,
    HMACSigner,
    MeshNode,
    MeshTask,
    RedisMeshTransport,
    ReplayConfig,
    ReplayProtector,
)


def _handler(name: str):
    def _run(task: MeshTask) -> str:
        return json.dumps({
            "worker": name,
            "task_id": task.task_id,
            "capability": task.capability,
            "prompt": task.prompt,
            "ok": True,
        })

    return _run


def main():
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    hmac_secret = os.environ.get("MESH_HMAC_SECRET", "")
    if not hmac_secret:
        raise RuntimeError("Set MESH_HMAC_SECRET with a shared random key.")

    if RedisMeshTransport is None:
        raise RuntimeError("Redis transport unavailable. Install redis dependency.")

    signer = HMACSigner(secret=hmac_secret, key_id="demo_key")
    replay = ReplayProtector(ReplayConfig(max_age_seconds=180.0, max_skew_seconds=20.0))

    transport = RedisMeshTransport(
        redis_url=redis_url,
        channel_prefix="jade:secure:mesh:",
        tls=redis_url.startswith("rediss://"),
        signer=signer,
        replay_protector=replay,
        poll_timeout=0.05,
    )

    router = DistributedMeshRouter(
        redis_url=redis_url,
        registry_prefix="jade:secure:mesh:registry",
        stale_after=60.0,
        heartbeat_ttl=180,
        refresh_interval=0.1,
        tls=redis_url.startswith("rediss://"),
    )
    coordinator = MeshNode(
        node_id="coord_secure",
        capabilities={"orchestrate"},
        router=router,
        bus=transport,
        task_handler=_handler("coord_secure"),
        verbose=True,
    )
    MeshNode(
        node_id="worker_alpha",
        capabilities={"inspect"},
        router=router,
        bus=transport,
        task_handler=_handler("worker_alpha"),
    )
    MeshNode(
        node_id="worker_beta",
        capabilities={"inspect"},
        router=router,
        bus=transport,
        task_handler=_handler("worker_beta"),
    )

    task = MeshTask(
        capability="inspect",
        prompt="Inspect sector 7 and report anomalies.",
        requester="coord_secure",
        ttl=6,
    )
    task_id = coordinator.submit_task(task)

    deadline = time.time() + 5.0
    while time.time() < deadline:
        coordinator.step()
        time.sleep(0.05)
        result = coordinator.get_result(task_id)
        if result is not None:
            print("Task result:", result.output)
            break
    else:
        print("Timed out waiting for result.")

    transport.close()


if __name__ == "__main__":
    main()
