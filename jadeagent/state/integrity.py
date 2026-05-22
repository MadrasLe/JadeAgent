"""Integrity and secret-redaction helpers for JGX capsules."""

from __future__ import annotations

import re
from typing import Any

from .artifact import JadeExecutionCapsule
from .manifest import JGX_FORMAT, JGX_MAGIC, canonical_json_hash


SECRET_PATTERNS = (
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]+"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret)\s*[:=]\s*['\"]?[^'\"\s,}]+"),
)


def _redact_string(value: str) -> str:
    redacted = value
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def redact_secrets(value: Any) -> Any:
    """Return a JSON-compatible copy with likely secrets replaced."""

    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, dict):
        return {str(key): redact_secrets(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [redact_secrets(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return redact_secrets(to_dict())
    return value


def find_secret_paths(value: Any, *, path: str = "$") -> list[str]:
    """Return JSON-ish paths where values look like secrets.

    The returned values intentionally contain only paths, never the matching
    secret text itself.
    """

    paths: list[str] = []
    if isinstance(value, str):
        if any(pattern.search(value) for pattern in SECRET_PATTERNS):
            paths.append(path)
        return paths
    if isinstance(value, dict):
        for key, item in value.items():
            paths.extend(find_secret_paths(item, path=f"{path}.{key}"))
        return paths
    if isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            paths.extend(find_secret_paths(item, path=f"{path}[{index}]"))
        return paths
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return find_secret_paths(to_dict(), path=path)
    return paths


def event_chain_hash(events: list[Any]) -> str:
    """Compute a deterministic chained hash over event order."""

    previous = ""
    for index, event in enumerate(events):
        data = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        previous = canonical_json_hash({
            "index": index,
            "previous": previous,
            "event": redact_secrets(data),
        })
    return previous


def snapshot_hashes(snapshots: list[Any]) -> list[dict[str, str]]:
    """Return stable hashes for snapshots without exposing secret values."""

    rows: list[dict[str, str]] = []
    for snapshot in snapshots:
        data = snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(snapshot)
        rows.append({
            "snapshot_id": str(data.get("snapshot_id", "")),
            "phase": str(data.get("phase", "")),
            "hash": canonical_json_hash(redact_secrets(data)),
        })
    return rows


def verify_capsule(capsule: JadeExecutionCapsule) -> dict[str, Any]:
    """Return a verification report for a JGX capsule."""

    issues: list[str] = []
    manifest = capsule.manifest
    latest = capsule.latest_snapshot
    snapshot_ids = {snapshot.snapshot_id for snapshot in capsule.snapshots}

    if manifest.magic != JGX_MAGIC:
        issues.append(f"unexpected magic: {manifest.magic}")
    if manifest.format != JGX_FORMAT:
        issues.append(f"unexpected format: {manifest.format}")
    if manifest.latest_snapshot_id and manifest.latest_snapshot_id not in snapshot_ids:
        issues.append("manifest latest_snapshot_id does not exist in snapshots")
    if capsule.snapshots and latest is None:
        issues.append("capsule has snapshots but no latest snapshot")

    previous_timestamp = None
    for event in capsule.events:
        if previous_timestamp is not None and event.timestamp < previous_timestamp:
            issues.append("event timestamps are not monotonic in stored order")
            break
        previous_timestamp = event.timestamp

    manifest_secret_paths = find_secret_paths(manifest.to_dict(), path="manifest")
    event_secret_paths: list[str] = []
    for index, event in enumerate(capsule.events):
        event_secret_paths.extend(find_secret_paths(event.to_dict(), path=f"events[{index}]"))
    snapshot_secret_paths: list[str] = []
    for index, snapshot in enumerate(capsule.snapshots):
        snapshot_secret_paths.extend(find_secret_paths(snapshot.to_dict(), path=f"snapshots[{index}]"))
    secret_paths = manifest_secret_paths + event_secret_paths + snapshot_secret_paths

    if secret_paths:
        issues.append(f"possible secret material found in {len(secret_paths)} field(s)")

    snapshot_rows = snapshot_hashes(capsule.snapshots)
    return {
        "ok": not issues,
        "run_id": manifest.run_id,
        "magic": manifest.magic,
        "format": manifest.format,
        "schema_version": manifest.schema_version,
        "event_count": len(capsule.events),
        "snapshot_count": len(capsule.snapshots),
        "latest_snapshot_id": latest.snapshot_id if latest is not None else "",
        "latest_phase": latest.phase if latest is not None else "",
        "event_chain_hash": event_chain_hash(capsule.events),
        "snapshot_hash": canonical_json_hash(snapshot_rows),
        "snapshot_hashes": snapshot_rows,
        "secret_leak_count": len(secret_paths),
        "secret_paths": secret_paths,
        "issues": issues,
    }
