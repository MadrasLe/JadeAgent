"""
Optional E2B-backed sandbox provider.

This module is intentionally defensive because E2B SDKs evolved over time.
It supports the common APIs from both `e2b_code_interpreter` and `e2b`.
"""

from __future__ import annotations

import inspect
import json
import os
import textwrap
from contextlib import contextmanager
from typing import Any

from .base import SandboxProvider, SandboxRunRequest, SandboxRunResult


def _truncate(value: str, max_chars: int = 200_000) -> str:
    text = value or ""
    if len(text) <= max_chars:
        return text
    suffix = f"\n...[truncated {len(text) - max_chars} chars]"
    return text[:max_chars] + suffix


class E2BSandboxProvider(SandboxProvider):
    """Execute code/commands in an E2B sandbox."""

    def __init__(
        self,
        api_key: str | None = None,
        template: str | None = None,
        keep_alive_seconds: int | None = None,
        reuse_session: bool = True,
    ):
        sandbox_cls = self._load_sandbox_class()
        if sandbox_cls is None:
            raise ImportError(
                "E2B SDK not found. Install with: pip install e2b-code-interpreter"
            )

        self._sandbox_cls = sandbox_cls
        self.api_key = api_key
        self.template = template
        self.keep_alive_seconds = keep_alive_seconds
        self.reuse_session = bool(reuse_session)
        self._sandbox: Any | None = None

    @property
    def name(self) -> str:
        return "e2b"

    def run(self, request: SandboxRunRequest) -> SandboxRunResult:
        result = SandboxRunResult(
            success=False,
            provider=self.name,
            metadata={
                "mode": request.mode,
                "template": self.template,
                **dict(request.metadata),
            },
        )

        mode = request.mode.strip().lower()
        if mode not in {"python", "shell"}:
            result.finalize(
                success=False,
                exit_code=2,
                error=f"Unsupported sandbox mode '{request.mode}'.",
            )
            return result

        sandbox = None
        try:
            sandbox = self._ensure_sandbox()
            if mode == "python":
                run_output = self._run_python(
                    sandbox=sandbox,
                    code=request.content,
                    timeout_seconds=request.timeout_seconds,
                )
            else:
                run_output = self._run_shell(
                    sandbox=sandbox,
                    command=request.content,
                    timeout_seconds=request.timeout_seconds,
                )

            result.finalize(
                success=bool(run_output.get("success", False)),
                exit_code=int(run_output.get("exit_code", 0)),
                stdout=_truncate(str(run_output.get("stdout", ""))),
                stderr=_truncate(str(run_output.get("stderr", ""))),
                error=run_output.get("error"),
            )
            result.metadata.update({
                "sandbox_id": run_output.get("sandbox_id"),
            })
            if run_output.get("raw") is not None:
                result.metadata["raw"] = run_output["raw"]
        except Exception as exc:
            result.finalize(success=False, exit_code=1, error=str(exc))
        finally:
            if not self.reuse_session and sandbox is not None:
                self._close_sandbox(sandbox)
                self._sandbox = None

        return result

    def close(self):
        if self._sandbox is not None:
            self._close_sandbox(self._sandbox)
            self._sandbox = None

    def _ensure_sandbox(self) -> Any:
        if self.reuse_session and self._sandbox is not None:
            return self._sandbox

        sandbox = self._create_sandbox(self._sandbox_cls)
        if self.reuse_session:
            self._sandbox = sandbox
        return sandbox

    def _build_create_kwargs(self, factory: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        try:
            params = inspect.signature(factory).parameters
        except Exception:
            params = {}

        if self.api_key and "api_key" in params:
            kwargs["api_key"] = self.api_key
        if self.keep_alive_seconds is not None:
            if "timeout" in params:
                kwargs["timeout"] = self.keep_alive_seconds
            elif "keep_alive_seconds" in params:
                kwargs["keep_alive_seconds"] = self.keep_alive_seconds
            elif "timeout_seconds" in params:
                kwargs["timeout_seconds"] = self.keep_alive_seconds
            elif "timeoutMs" in params:
                kwargs["timeoutMs"] = int(self.keep_alive_seconds * 1000)
        return kwargs

    def _create_sandbox(self, sandbox_cls: Any) -> Any:
        create_fn = getattr(sandbox_cls, "create", None)
        if callable(create_fn):
            kwargs = self._build_create_kwargs(create_fn)
            with self._api_key_env():
                if self.template:
                    try:
                        return create_fn(template=self.template, **kwargs)
                    except TypeError:
                        return create_fn(self.template, **kwargs)
                return create_fn(**kwargs)

        kwargs = self._build_create_kwargs(sandbox_cls)
        if self.template:
            kwargs["template"] = self.template
        with self._api_key_env():
            return sandbox_cls(**kwargs)

    @contextmanager
    def _api_key_env(self):
        if not self.api_key:
            yield
            return

        existing = os.environ.get("E2B_API_KEY")
        os.environ["E2B_API_KEY"] = self.api_key
        try:
            yield
        finally:
            if existing is None:
                os.environ.pop("E2B_API_KEY", None)
            else:
                os.environ["E2B_API_KEY"] = existing

    @staticmethod
    def _close_sandbox(sandbox: Any):
        kill_fn = getattr(sandbox, "kill", None)
        if callable(kill_fn):
            try:
                kill_fn()
                return
            except Exception:
                pass

        close_fn = getattr(sandbox, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    @staticmethod
    def _load_sandbox_class():
        try:
            from e2b_code_interpreter import Sandbox as SandboxClass
            return SandboxClass
        except Exception:
            pass

        try:
            from e2b import Sandbox as SandboxClass
            return SandboxClass
        except Exception:
            pass

        return None

    def _run_python(self, sandbox: Any, code: str, timeout_seconds: float) -> dict[str, Any]:
        run_code = getattr(sandbox, "run_code", None)
        if not callable(run_code):
            raise RuntimeError("E2B sandbox does not expose run_code().")

        kwargs = self._supported_kwargs(run_code, {
            "timeout": max(float(timeout_seconds), 0.1),
        })
        execution = run_code(code, **kwargs)
        parsed = self._parse_execution(execution)
        parsed["sandbox_id"] = getattr(sandbox, "sandbox_id", None) or getattr(sandbox, "id", None)
        return parsed

    def _run_shell(self, sandbox: Any, command: str, timeout_seconds: float) -> dict[str, Any]:
        commands_api = getattr(sandbox, "commands", None)
        run_fn = getattr(commands_api, "run", None) if commands_api is not None else None

        if callable(run_fn):
            kwargs = self._supported_kwargs(run_fn, {
                "timeout": max(float(timeout_seconds), 0.1),
            })
            execution = run_fn(command, **kwargs)
            parsed = self._parse_execution(execution)
            parsed["sandbox_id"] = getattr(sandbox, "sandbox_id", None) or getattr(sandbox, "id", None)
            return parsed

        sentinel = "__JADEAGENT_E2B__"
        safe_code = textwrap.dedent(
            f"""
            import json
            import subprocess

            cmd = {command!r}
            try:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout={max(float(timeout_seconds), 0.1)!r},
                )
                payload = {{
                    "exit_code": int(proc.returncode),
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }}
            except subprocess.TimeoutExpired as exc:
                payload = {{
                    "exit_code": 124,
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                    "error": "timeout",
                }}
            print("{sentinel}" + json.dumps(payload, separators=(",", ":"), ensure_ascii=True))
            """
        ).strip()

        parsed = self._run_python(sandbox=sandbox, code=safe_code, timeout_seconds=timeout_seconds)
        decoded = self._decode_shell_fallback(parsed.get("stdout", ""), sentinel=sentinel)
        if decoded is None:
            return parsed

        exit_code = int(decoded.get("exit_code", 1))
        stdout = str(decoded.get("stdout", ""))
        stderr = str(decoded.get("stderr", ""))
        error = decoded.get("error")
        return {
            "success": exit_code == 0 and not error,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "error": "Sandbox command failed." if exit_code != 0 and not error else error,
            "raw": decoded,
            "sandbox_id": parsed.get("sandbox_id"),
        }

    @staticmethod
    def _decode_shell_fallback(stdout: str, sentinel: str) -> dict[str, Any] | None:
        lines = (stdout or "").splitlines()
        for idx in range(len(lines) - 1, -1, -1):
            line = lines[idx].strip()
            if not line.startswith(sentinel):
                continue
            payload_text = line[len(sentinel):]
            try:
                return json.loads(payload_text)
            except Exception:
                return None
        return None

    @staticmethod
    def _parse_execution(execution: Any) -> dict[str, Any]:
        stdout = ""
        stderr = ""
        raw_error = None
        exit_code = 0

        if execution is None:
            return {
                "success": True,
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "error": None,
                "raw": None,
            }

        raw_error = getattr(execution, "error", None)

        logs = getattr(execution, "logs", None)
        if logs is not None:
            if isinstance(logs, dict):
                stdout = E2BSandboxProvider._coerce_text(logs.get("stdout", ""))
                stderr = E2BSandboxProvider._coerce_text(logs.get("stderr", ""))
            else:
                stdout = E2BSandboxProvider._coerce_text(getattr(logs, "stdout", ""))
                stderr = E2BSandboxProvider._coerce_text(getattr(logs, "stderr", ""))

        if not stdout:
            stdout = E2BSandboxProvider._coerce_text(getattr(execution, "stdout", ""))
        if not stderr:
            stderr = E2BSandboxProvider._coerce_text(getattr(execution, "stderr", ""))

        code_candidate = getattr(execution, "exit_code", None)
        if code_candidate is None:
            code_candidate = getattr(execution, "returncode", None)
        if code_candidate is not None:
            try:
                exit_code = int(code_candidate)
            except Exception:
                exit_code = 1
        elif raw_error:
            exit_code = 1

        error_text = None
        if raw_error:
            error_text = E2BSandboxProvider._coerce_text(raw_error)
            if not error_text:
                error_text = "Sandbox execution failed."

        return {
            "success": exit_code == 0 and error_text is None,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "error": error_text,
            "raw": E2BSandboxProvider._safe_repr(execution),
        }

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (bytes, bytearray)):
            return bytes(value).decode("utf-8", errors="replace")
        if isinstance(value, list):
            return "".join(E2BSandboxProvider._coerce_text(item) for item in value)
        text_attr = getattr(value, "text", None)
        if isinstance(text_attr, str):
            return text_attr
        return str(value)

    @staticmethod
    def _safe_repr(value: Any) -> dict[str, Any]:
        try:
            if hasattr(value, "__dict__"):
                raw = dict(value.__dict__)
                return {k: str(v) for k, v in raw.items()}
        except Exception:
            pass
        return {"repr": repr(value)}

    @staticmethod
    def _supported_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            params = inspect.signature(func).parameters
        except Exception:
            return {}
        return {k: v for k, v in kwargs.items() if k in params}
