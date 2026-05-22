"""
Base models and interfaces for sandbox execution providers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class SandboxRunRequest:
    """
    Unified request format for sandbox execution.

    `mode` values:
    - "shell": execute shell command in sandbox/container.
    - "python": execute Python code in sandbox/container.
    """

    mode: str
    content: str
    timeout_seconds: float = 30.0
    workdir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxRunResult:
    """Execution result returned by sandbox providers."""

    success: bool
    provider: str
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def finalize(
        self,
        success: bool,
        exit_code: int,
        stdout: str = "",
        stderr: str = "",
        error: str | None = None,
    ):
        self.success = success
        self.exit_code = int(exit_code)
        self.stdout = stdout
        self.stderr = stderr
        self.error = error
        self.finished_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "provider": self.provider,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metadata": self.metadata,
        }


class SandboxProvider(Protocol):
    """Interface for sandbox execution backends."""

    @property
    def name(self) -> str:
        ...

    def run(self, request: SandboxRunRequest) -> SandboxRunResult:
        ...

