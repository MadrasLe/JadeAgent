"""
Helpers to execute mesh tasks in sandbox providers.
"""

from __future__ import annotations

import json
from typing import Any

from ..mesh.protocol import MeshTask
from .base import SandboxProvider, SandboxRunRequest


def parse_sandbox_request(
    task: MeshTask,
    default_mode: str = "python",
    default_timeout_seconds: float = 30.0,
    allow_prompt_fallback: bool = True,
) -> SandboxRunRequest:
    """
    Parse a MeshTask into SandboxRunRequest.

    Preferred task format:
    - task.metadata["sandbox"] = {
        "mode": "python" | "shell",
        "content": "...",
        "timeout_seconds": 20,
        "workdir": "/workspace",
        ...
      }

    Fallback prompt formats:
    - "python:<code>"
    - "shell:<command>"
    """

    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    sandbox_meta = metadata.get("sandbox", {})
    if not isinstance(sandbox_meta, dict):
        sandbox_meta = {}

    mode = str(sandbox_meta.get("mode") or default_mode).strip().lower()
    content = sandbox_meta.get("content")
    timeout_seconds = sandbox_meta.get("timeout_seconds", default_timeout_seconds)
    workdir = sandbox_meta.get("workdir")

    if content is None and allow_prompt_fallback:
        prompt = task.prompt or ""
        if prompt.startswith("python:"):
            mode = "python"
            content = prompt[len("python:"):].lstrip()
        elif prompt.startswith("shell:"):
            mode = "shell"
            content = prompt[len("shell:"):].lstrip()
        else:
            content = prompt

    content_text = str(content or "")
    try:
        timeout_float = float(timeout_seconds)
    except Exception:
        timeout_float = float(default_timeout_seconds)

    request_metadata = {
        "task_id": task.task_id,
        "capability": task.capability,
        "requester": task.requester,
        "mesh_metadata": metadata,
    }
    if "extra" in sandbox_meta:
        request_metadata["extra"] = sandbox_meta["extra"]

    return SandboxRunRequest(
        mode=mode,
        content=content_text,
        timeout_seconds=max(timeout_float, 0.1),
        workdir=str(workdir) if workdir is not None else None,
        metadata=request_metadata,
    )


def make_sandbox_task_handler(
    node_id: str,
    provider: SandboxProvider,
    default_mode: str = "python",
    default_timeout_seconds: float = 30.0,
) -> callable:
    """Create a Mesh task_handler backed by a sandbox provider."""

    def _run(task: MeshTask) -> str:
        request = parse_sandbox_request(
            task=task,
            default_mode=default_mode,
            default_timeout_seconds=default_timeout_seconds,
            allow_prompt_fallback=True,
        )
        run_result = provider.run(request)

        payload: dict[str, Any] = {
            "worker": node_id,
            "task_id": task.task_id,
            "capability": task.capability,
            "prompt": task.prompt,
            "success": run_result.success,
            "sandbox": run_result.to_dict(),
        }

        if not run_result.success:
            error_payload = {
                "worker": node_id,
                "task_id": task.task_id,
                "capability": task.capability,
                "provider": run_result.provider,
                "exit_code": run_result.exit_code,
                "error": run_result.error or "Sandbox execution failed.",
                "stderr": run_result.stderr,
                "stdout": run_result.stdout,
                "sandbox_id": run_result.metadata.get("sandbox_id"),
            }
            raise RuntimeError(json.dumps(error_payload, ensure_ascii=True))

        return json.dumps(payload, ensure_ascii=True)

    return _run
