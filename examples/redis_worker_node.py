"""
Run a persistent distributed mesh worker over secure Redis transport.

Usage:
    set REDIS_URL=rediss://<host>:<port>/0
    set MESH_HMAC_SECRET=<shared-secret>
    set MESH_NODE_ID=worker_alpha
    set MESH_CAPABILITIES=inspect,scan
    python examples/redis_worker_node.py
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


def _task_handler(node_id: str):
    def _run(task: MeshTask) -> str:
        payload = {
            "worker": node_id,
            "task_id": task.task_id,
            "capability": task.capability,
            "prompt": task.prompt,
            "metadata": task.metadata,
            "success": True,
        }
        return json.dumps(payload)

    return _run


def main():
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    secret = os.environ.get("MESH_HMAC_SECRET", "")
    node_id = os.environ.get("MESH_NODE_ID", "worker_default")
    capabilities_raw = os.environ.get("MESH_CAPABILITIES", "inspect")

    if not secret:
        raise RuntimeError("Set MESH_HMAC_SECRET with a shared key.")

    capabilities = {
        token.strip()
        for token in capabilities_raw.split(",")
        if token.strip()
    }
    if not capabilities:
        capabilities = {"inspect"}

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

    node = MeshNode(
        node_id=node_id,
        capabilities=capabilities,
        router=router,
        bus=transport,
        task_handler=_task_handler(node_id),
        verbose=False,
    )

    print(f"[worker:{node_id}] online with capabilities={sorted(capabilities)}")
    try:
        while True:
            node.step()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print(f"\n[worker:{node_id}] shutting down")
    finally:
        try:
            router.unregister_node(node_id)
        except Exception:
            pass
        transport.close()


if __name__ == "__main__":
    main()

