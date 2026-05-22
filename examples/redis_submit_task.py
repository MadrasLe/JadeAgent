"""
Submit a task to the distributed mesh and wait for result.

Usage:
    set REDIS_URL=rediss://<host>:<port>/0
    set MESH_HMAC_SECRET=<shared-secret>
    python examples/redis_submit_task.py inspect "Inspect sector 7 now"
"""

from __future__ import annotations

import os
import sys
import time
import uuid
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


def main():
    capability = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Inspect sector 7 and report."
    timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    secret = os.environ.get("MESH_HMAC_SECRET", "")
    if not secret:
        raise RuntimeError("Set MESH_HMAC_SECRET with a shared key.")

    coordinator_id = f"coord_cli_{uuid.uuid4().hex[:8]}"

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

    coordinator = MeshNode(
        node_id=coordinator_id,
        capabilities={"orchestrate"},
        router=router,
        bus=transport,
        task_handler=lambda task: f"coord ack {task.task_id}",
        verbose=False,
    )

    task = MeshTask(
        capability=capability,
        prompt=prompt,
        requester=coordinator_id,
        ttl=8,
    )
    task_id = coordinator.submit_task(task)
    print(f"[submit] task_id={task_id} capability={capability}")

    deadline = time.time() + timeout
    while time.time() < deadline:
        coordinator.step()
        result = coordinator.get_result(task_id)
        if result is not None:
            if result.output:
                print("[submit] result:", result.output)
            else:
                print(
                    "[submit] result error:",
                    {
                        "state": result.state.value,
                        "error": result.error,
                        "task_id": result.task_id,
                        "capability": result.capability,
                        "node_id": result.node_id,
                    },
                )
            break
        time.sleep(0.05)
    else:
        print("[submit] timeout waiting for result")

    try:
        router.unregister_node(coordinator_id)
    except Exception:
        pass
    transport.close()


if __name__ == "__main__":
    main()
