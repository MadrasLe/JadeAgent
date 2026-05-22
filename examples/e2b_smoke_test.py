"""
Direct E2B provider smoke test without Redis/mesh.

Usage:
    set E2B_API_KEY=e2b_xxx
    python examples/e2b_smoke_test.py

Optional:
    set E2B_TEMPLATE=<template-id>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jadeagent.sandbox import E2BSandboxProvider, SandboxRunRequest


def _print_result(label: str, result):
    print(f"\n=== {label} ===")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


def main():
    api_key = os.environ.get("E2B_API_KEY")
    template = os.environ.get("E2B_TEMPLATE")

    if not api_key:
        raise RuntimeError("Set E2B_API_KEY before running this script.")
    if E2BSandboxProvider is None:
        raise RuntimeError("E2B provider unavailable. Install e2b-code-interpreter.")

    provider = E2BSandboxProvider(
        api_key=api_key,
        template=template,
        keep_alive_seconds=300,
        reuse_session=True,
    )

    try:
        python_result = provider.run(
            SandboxRunRequest(
                mode="python",
                content="print('hello from e2b'); print(6*7)",
                timeout_seconds=20.0,
            )
        )
        _print_result("PYTHON", python_result)

        shell_result = provider.run(
            SandboxRunRequest(
                mode="shell",
                content="echo inside_e2b && uname -a",
                timeout_seconds=20.0,
            )
        )
        _print_result("SHELL", shell_result)
    finally:
        provider.close()


if __name__ == "__main__":
    main()
