"""
Persistent distributed mesh worker that executes sandbox tasks.

Usage (subprocess provider):
    set REDIS_URL=redis://localhost:6379/0
    set MESH_HMAC_SECRET=my_secret
    set MESH_NODE_ID=sandbox_worker_a
    set MESH_CAPABILITIES=sandbox_exec
    set SANDBOX_PROVIDER=subprocess
    python examples/redis_sandbox_worker.py

Usage (E2B provider):
    set REDIS_URL=rediss://<host>:<port>/0
    set MESH_HMAC_SECRET=my_secret
    set SANDBOX_PROVIDER=e2b
    set E2B_API_KEY=e2b_xxx
    python examples/redis_sandbox_worker.py
"""

from __future__ import annotations

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
    RedisMeshTransport,
    ReplayConfig,
    ReplayProtector,
)
from jadeagent.sandbox import (
    E2BSandboxProvider,
    SubprocessSandboxProvider,
    make_sandbox_task_handler,
)


def _as_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _build_provider():
    provider_name = os.environ.get("SANDBOX_PROVIDER", "subprocess").strip().lower()

    if provider_name == "subprocess":
        allow_shell = _as_bool(os.environ.get("SANDBOX_ALLOW_SHELL"), default=True)
        python_executable = os.environ.get("SANDBOX_PYTHON_EXECUTABLE", "python")
        return SubprocessSandboxProvider(
            allow_shell_mode=allow_shell,
            python_executable=python_executable,
        )

    if provider_name == "e2b":
        if E2BSandboxProvider is None:
            raise RuntimeError(
                "E2B provider unavailable. Install optional deps: pip install e2b-code-interpreter"
            )
        keep_alive = os.environ.get("E2B_KEEP_ALIVE_SECONDS")
        keep_alive_seconds = int(keep_alive) if keep_alive else None
        reuse_session = _as_bool(os.environ.get("E2B_REUSE_SESSION"), default=True)
        return E2BSandboxProvider(
            api_key=os.environ.get("E2B_API_KEY"),
            template=os.environ.get("E2B_TEMPLATE"),
            keep_alive_seconds=keep_alive_seconds,
            reuse_session=reuse_session,
        )

    raise RuntimeError(f"Unsupported SANDBOX_PROVIDER={provider_name!r}.")


def main():
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    secret = os.environ.get("MESH_HMAC_SECRET", "")
    node_id = os.environ.get("MESH_NODE_ID", "sandbox_worker_default")
    capabilities_raw = os.environ.get("MESH_CAPABILITIES", "sandbox_exec")

    if not secret:
        raise RuntimeError("Set MESH_HMAC_SECRET with a shared key.")

    capabilities = {token.strip() for token in capabilities_raw.split(",") if token.strip()}
    if not capabilities:
        capabilities = {"sandbox_exec"}

    default_mode = os.environ.get("SANDBOX_DEFAULT_MODE", "python").strip().lower()
    default_timeout = float(os.environ.get("SANDBOX_DEFAULT_TIMEOUT", "30"))

    provider = _build_provider()
    task_handler = make_sandbox_task_handler(
        node_id=node_id,
        provider=provider,
        default_mode=default_mode,
        default_timeout_seconds=default_timeout,
    )

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
        task_handler=task_handler,
        verbose=False,
    )

    provider_name = getattr(provider, "name", type(provider).__name__)
    print(
        f"[sandbox-worker:{node_id}] online provider={provider_name} "
        f"caps={sorted(capabilities)} default_mode={default_mode}"
    )

    try:
        while True:
            node.step()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print(f"\n[sandbox-worker:{node_id}] shutting down")
    finally:
        try:
            router.unregister_node(node_id)
        except Exception:
            pass
        close_fn = getattr(provider, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
        transport.close()


if __name__ == "__main__":
    main()

