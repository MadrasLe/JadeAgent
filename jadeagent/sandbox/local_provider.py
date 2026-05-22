"""
Local subprocess-based sandbox provider.

This provider is useful for development and on-prem workers where each node
owns an isolated environment/container.
"""

from __future__ import annotations

import subprocess
from typing import Any

from .base import SandboxProvider, SandboxRunRequest, SandboxRunResult


class SubprocessSandboxProvider(SandboxProvider):
    """Execute tasks using local subprocess commands."""

    def __init__(
        self,
        default_shell: str = "bash",
        allow_shell_mode: bool = True,
        python_executable: str = "python",
    ):
        self.default_shell = default_shell
        self.allow_shell_mode = allow_shell_mode
        self.python_executable = python_executable

    @property
    def name(self) -> str:
        return "subprocess"

    def run(self, request: SandboxRunRequest) -> SandboxRunResult:
        result = SandboxRunResult(
            success=False,
            provider=self.name,
            metadata={"mode": request.mode, **dict(request.metadata)},
        )

        mode = request.mode.strip().lower()
        if mode == "python":
            cmd = [self.python_executable, "-c", request.content]
            shell = False
        elif mode == "shell":
            if not self.allow_shell_mode:
                result.finalize(
                    success=False,
                    exit_code=126,
                    error="Shell mode is disabled for this provider.",
                )
                return result
            cmd = request.content
            shell = True
        else:
            result.finalize(
                success=False,
                exit_code=2,
                error=f"Unsupported sandbox mode '{request.mode}'.",
            )
            return result

        try:
            completed = subprocess.run(
                cmd,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=max(float(request.timeout_seconds), 0.1),
                cwd=request.workdir,
            )
            result.finalize(
                success=(completed.returncode == 0),
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                error=None if completed.returncode == 0 else "Sandbox command failed.",
            )
        except subprocess.TimeoutExpired as exc:
            result.finalize(
                success=False,
                exit_code=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                error=f"Timeout after {request.timeout_seconds}s.",
            )
        except Exception as exc:
            result.finalize(
                success=False,
                exit_code=1,
                error=str(exc),
            )

        return result

