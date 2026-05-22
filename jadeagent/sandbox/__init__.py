"""Sandbox execution providers and mesh integration helpers."""

from .base import SandboxProvider, SandboxRunRequest, SandboxRunResult
from .local_provider import SubprocessSandboxProvider
from .mesh_handler import parse_sandbox_request, make_sandbox_task_handler

try:
    from .e2b_provider import E2BSandboxProvider
except Exception:  # pragma: no cover - optional dependency
    E2BSandboxProvider = None

__all__ = [
    "SandboxProvider",
    "SandboxRunRequest",
    "SandboxRunResult",
    "SubprocessSandboxProvider",
    "E2BSandboxProvider",
    "parse_sandbox_request",
    "make_sandbox_task_handler",
]

