"""Restore compatibility checks for governed execution capsules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .manifest import JGX_MAGIC, JGX_SCHEMA_VERSION, JadeStateManifest


@dataclass
class CompatibilityReport:
    allowed: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def require_allowed(self) -> None:
        if not self.allowed:
            raise ValueError("; ".join(self.issues) or "state restore is not compatible")


def _check_expected(
    issues: list[str],
    label: str,
    actual: str,
    expected: str | None,
    *,
    required: bool = True,
) -> None:
    if expected is None:
        return
    if required and not actual:
        issues.append(f"{label} is missing from manifest")
        return
    if actual and expected and actual != expected:
        issues.append(f"{label} mismatch: manifest={actual!r} expected={expected!r}")


def validate_restore_compatibility(
    manifest: JadeStateManifest,
    *,
    tenant_id: str | None = None,
    policy_hash: str | None = None,
    tool_registry_hash: str | None = None,
    memory_scope_hash: str | None = None,
    model_fingerprint: str | None = None,
    backend: str | None = None,
    allow_schema_minor_mismatch: bool = True,
    allow_policy_migration: bool = False,
    allow_tool_registry_migration: bool = False,
    extra: dict[str, Any] | None = None,
) -> CompatibilityReport:
    """Validate whether a capsule may be restored in the current runtime."""

    issues: list[str] = []
    warnings: list[str] = []

    if manifest.magic != JGX_MAGIC:
        issues.append(f"unsupported magic: {manifest.magic!r}")

    if manifest.schema_version != JGX_SCHEMA_VERSION:
        if allow_schema_minor_mismatch and manifest.schema_version.split(".")[0] == JGX_SCHEMA_VERSION.split(".")[0]:
            warnings.append(
                f"schema version differs: manifest={manifest.schema_version!r} runtime={JGX_SCHEMA_VERSION!r}"
            )
        else:
            issues.append(
                f"schema version mismatch: manifest={manifest.schema_version!r} runtime={JGX_SCHEMA_VERSION!r}"
            )

    _check_expected(issues, "tenant_id", manifest.tenant_id, tenant_id, required=False)
    _check_expected(issues, "memory_scope_hash", manifest.memory_scope_hash, memory_scope_hash, required=False)
    _check_expected(issues, "model_fingerprint", manifest.model_fingerprint, model_fingerprint, required=False)
    _check_expected(issues, "backend", manifest.backend, backend, required=False)

    if policy_hash is not None and manifest.policy_hash and manifest.policy_hash != policy_hash:
        message = f"policy_hash mismatch: manifest={manifest.policy_hash!r} expected={policy_hash!r}"
        if allow_policy_migration:
            warnings.append(message)
        else:
            issues.append(message)

    if tool_registry_hash is not None and manifest.tool_registry_hash and manifest.tool_registry_hash != tool_registry_hash:
        message = (
            "tool_registry_hash mismatch: "
            f"manifest={manifest.tool_registry_hash!r} expected={tool_registry_hash!r}"
        )
        if allow_tool_registry_migration:
            warnings.append(message)
        else:
            issues.append(message)

    for key, expected_value in (extra or {}).items():
        actual_value = manifest.metadata.get(key)
        if actual_value != expected_value:
            issues.append(f"metadata[{key!r}] mismatch: manifest={actual_value!r} expected={expected_value!r}")

    return CompatibilityReport(allowed=not issues, issues=issues, warnings=warnings)
