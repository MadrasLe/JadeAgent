"""
Submit a sandbox task to distributed mesh and wait for result.

Usage:
    set REDIS_URL=redis://localhost:6379/0
    set MESH_HMAC_SECRET=my_secret
    python examples/redis_sandbox_submit.py python "print('hello from mesh sandbox')"
    python examples/redis_sandbox_submit.py shell "echo hi from shell"
"""

from __future__ import annotations

import argparse
import json
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit distributed sandbox mesh task.")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["python", "shell"],
        default="python",
        help="Sandbox execution mode.",
    )
    parser.add_argument(
        "content",
        nargs="?",
        default="print('hello from sandbox mesh')",
        help="Code (python) or command (shell).",
    )
    parser.add_argument(
        "wait_timeout",
        nargs="?",
        type=float,
        default=15.0,
        help="Seconds to wait for task result.",
    )
    parser.add_argument(
        "--exec-timeout",
        type=float,
        default=20.0,
        help="Execution timeout sent to worker provider.",
    )
    parser.add_argument(
        "--capability",
        default=os.environ.get("MESH_SANDBOX_CAPABILITY", "sandbox_exec"),
        help="Capability tag used for routing.",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Optional remote working directory in sandbox.",
    )
    parser.add_argument(
        "--wait-route-seconds",
        type=float,
        default=10.0,
        help="Wait up to N seconds for an available route before submitting.",
    )
    parser.add_argument(
        "--route-poll-interval",
        type=float,
        default=0.2,
        help="Polling interval while waiting for route.",
    )
    return parser.parse_args()


def _has_route(router: DistributedMeshRouter, capability: str) -> bool:
    for row in router.snapshot():
        caps = set(row.get("capabilities", []))
        if capability in caps:
            return True
    return False


def main():
    args = _parse_args()

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    secret = os.environ.get("MESH_HMAC_SECRET", "")
    if not secret:
        raise RuntimeError("Set MESH_HMAC_SECRET with a shared key.")

    coordinator_id = f"sandbox_coord_{uuid.uuid4().hex[:8]}"

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

    route_available = _has_route(router, args.capability)
    if args.wait_route_seconds > 0 and not route_available:
        deadline = time.time() + float(args.wait_route_seconds)
        while time.time() < deadline:
            if _has_route(router, args.capability):
                route_available = True
                break
            time.sleep(max(float(args.route_poll_interval), 0.05))
    if not route_available:
        print(
            "[submit] no available workers for capability "
            f"{args.capability!r}. Aborting submit."
        )
        print("[submit] router snapshot:", json.dumps(router.snapshot(), indent=2))
        try:
            router.unregister_node(coordinator_id)
        except Exception:
            pass
        transport.close()
        return

    task = MeshTask(
        capability=args.capability,
        prompt=f"{args.mode}:{args.content}",
        requester=coordinator_id,
        ttl=8,
        metadata={
            "sandbox": {
                "mode": args.mode,
                "content": args.content,
                "timeout_seconds": float(args.exec_timeout),
                "workdir": args.workdir,
            }
        },
    )
    task_id = coordinator.submit_task(task)
    print(f"[submit] task_id={task_id} capability={args.capability}")

    deadline = time.time() + float(args.wait_timeout)
    while time.time() < deadline:
        coordinator.step()
        result = coordinator.get_result(task_id)
        if result is not None:
            if result.output:
                print("[submit] raw result:", result.output)
                try:
                    parsed = json.loads(result.output)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    print("[submit] parsed result:")
                    print(json.dumps(parsed, indent=2, ensure_ascii=False))
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
