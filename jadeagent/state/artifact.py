"""Read, write, and inspect Jade governed execution (.jgx) capsules."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .events import JadeStateEvent
from .manifest import JGX_MAGIC, JadeStateManifest
from .snapshot import AgentRuntimeSnapshot


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":")))
        handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@dataclass
class JadeExecutionCapsule:
    """In-memory representation of a .jgx capsule."""

    manifest: JadeStateManifest
    events: list[JadeStateEvent] = field(default_factory=list)
    snapshots: list[AgentRuntimeSnapshot] = field(default_factory=list)
    payloads: dict[str, bytes] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return self.manifest.run_id

    @property
    def latest_snapshot(self) -> AgentRuntimeSnapshot | None:
        if not self.snapshots:
            return None
        if self.manifest.latest_snapshot_id:
            for snapshot in reversed(self.snapshots):
                if snapshot.snapshot_id == self.manifest.latest_snapshot_id:
                    return snapshot
        return self.snapshots[-1]

    def to_directory(self, path: str | Path) -> Path:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        (root / "snapshots").mkdir(exist_ok=True)
        (root / "payloads").mkdir(exist_ok=True)

        for snapshot in self.snapshots:
            _write_json(root / "snapshots" / f"{snapshot.snapshot_id}.json", snapshot.to_dict())
            self.manifest.latest_snapshot_id = snapshot.snapshot_id
        self.manifest.touch()
        _write_json(root / "manifest.json", self.manifest.to_dict())

        events_path = root / "events.jsonl"
        if events_path.exists():
            events_path.unlink()
        for event in self.events:
            _append_jsonl(events_path, event.to_dict())

        for digest, payload in self.payloads.items():
            (root / "payloads" / digest).write_bytes(payload)

        return root

    @classmethod
    def from_directory(cls, path: str | Path) -> "JadeExecutionCapsule":
        root = Path(path)
        manifest = JadeStateManifest.from_dict(_read_json(root / "manifest.json"))
        snapshots = [
            AgentRuntimeSnapshot.from_dict(_read_json(snapshot_path))
            for snapshot_path in sorted((root / "snapshots").glob("*.json"))
        ]
        events = [
            JadeStateEvent.from_dict(row)
            for row in _read_jsonl(root / "events.jsonl")
        ]
        payloads = {
            payload_path.name: payload_path.read_bytes()
            for payload_path in sorted((root / "payloads").glob("*"))
            if payload_path.is_file()
        }
        return cls(manifest=manifest, events=events, snapshots=snapshots, payloads=payloads)

    def inspect(self) -> dict[str, Any]:
        latest = self.latest_snapshot
        return {
            "magic": self.manifest.magic,
            "format": self.manifest.format,
            "schema_version": self.manifest.schema_version,
            "run_id": self.manifest.run_id,
            "task_id": self.manifest.task_id,
            "agent_id": self.manifest.agent_id,
            "tenant_id": self.manifest.tenant_id,
            "capability": self.manifest.capability,
            "backend": self.manifest.backend,
            "event_count": len(self.events),
            "snapshot_count": len(self.snapshots),
            "latest_snapshot_id": latest.snapshot_id if latest is not None else "",
            "latest_phase": latest.phase if latest is not None else "",
            "latest_step": latest.step if latest is not None else 0,
            "payload_count": len(self.payloads),
        }


def write_jgx(path: str | Path, capsule: JadeExecutionCapsule) -> Path:
    """Write a .jgx directory capsule."""

    return capsule.to_directory(path)


def load_jgx(path: str | Path) -> JadeExecutionCapsule:
    """Load a .jgx directory capsule."""

    return JadeExecutionCapsule.from_directory(path)


def inspect_jgx(path: str | Path) -> dict[str, Any]:
    """Return lightweight metadata for a .jgx capsule."""

    return load_jgx(path).inspect()
