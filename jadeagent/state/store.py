"""StateStore implementations for Jade governed execution capsules."""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .artifact import (
    JadeExecutionCapsule,
    _append_jsonl,
    _read_json,
    _read_jsonl,
    _write_json,
)
from .events import JadeStateEvent
from .manifest import JadeStateManifest
from .snapshot import AgentRuntimeSnapshot


class StateStore(ABC):
    """Durable state-machine storage for agent runs."""

    @abstractmethod
    def create_run(self, manifest: JadeStateManifest) -> JadeStateManifest:
        ...

    @abstractmethod
    def append_event(self, run_id: str, event: JadeStateEvent | dict[str, Any]) -> JadeStateEvent:
        ...

    @abstractmethod
    def save_snapshot(self, run_id: str, snapshot: AgentRuntimeSnapshot) -> AgentRuntimeSnapshot:
        ...

    @abstractmethod
    def latest_snapshot(self, run_id: str) -> AgentRuntimeSnapshot | None:
        ...

    @abstractmethod
    def load_run(self, run_id: str) -> JadeExecutionCapsule:
        ...

    @abstractmethod
    def list_events(self, run_id: str, limit: int = 100) -> list[JadeStateEvent]:
        ...

    @abstractmethod
    def inspect(self, run_id: str) -> dict[str, Any]:
        ...


class InMemoryStateStore(StateStore):
    """Non-durable store useful for tests and embedded runtimes."""

    def __init__(self):
        self._manifests: dict[str, JadeStateManifest] = {}
        self._events: dict[str, list[JadeStateEvent]] = {}
        self._snapshots: dict[str, list[AgentRuntimeSnapshot]] = {}
        self._lock = threading.RLock()

    def create_run(self, manifest: JadeStateManifest) -> JadeStateManifest:
        with self._lock:
            self._manifests[manifest.run_id] = manifest
            self._events.setdefault(manifest.run_id, [])
            self._snapshots.setdefault(manifest.run_id, [])
            return manifest

    def append_event(self, run_id: str, event: JadeStateEvent | dict[str, Any]) -> JadeStateEvent:
        with self._lock:
            state_event = event if isinstance(event, JadeStateEvent) else JadeStateEvent.from_dict(event)
            state_event.run_id = state_event.run_id or run_id
            self._events.setdefault(run_id, []).append(state_event)
            manifest = self._manifests.get(run_id)
            if manifest is not None:
                manifest.touch()
            return state_event

    def save_snapshot(self, run_id: str, snapshot: AgentRuntimeSnapshot) -> AgentRuntimeSnapshot:
        with self._lock:
            self._snapshots.setdefault(run_id, []).append(snapshot)
            manifest = self._manifests.get(run_id)
            if manifest is not None:
                manifest.latest_snapshot_id = snapshot.snapshot_id
                manifest.touch()
            return snapshot

    def latest_snapshot(self, run_id: str) -> AgentRuntimeSnapshot | None:
        with self._lock:
            snapshots = self._snapshots.get(run_id, [])
            return snapshots[-1] if snapshots else None

    def load_run(self, run_id: str) -> JadeExecutionCapsule:
        with self._lock:
            manifest = self._manifests[run_id]
            return JadeExecutionCapsule(
                manifest=manifest,
                events=list(self._events.get(run_id, [])),
                snapshots=list(self._snapshots.get(run_id, [])),
            )

    def list_events(self, run_id: str, limit: int = 100) -> list[JadeStateEvent]:
        with self._lock:
            return list(self._events.get(run_id, [])[-limit:])

    def inspect(self, run_id: str) -> dict[str, Any]:
        return self.load_run(run_id).inspect()


class FileStateStore(StateStore):
    """Local filesystem .jgx store.

    Each run is a directory named ``<run_id>.jgx`` with manifest, events,
    snapshots, and payload folders. The layout is deliberately transparent so
    users can inspect state with normal filesystem tools.
    """

    def __init__(self, root: str | Path = ".jade_state"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _run_path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.jgx"

    def create_run(self, manifest: JadeStateManifest) -> JadeStateManifest:
        with self._lock:
            run_path = self._run_path(manifest.run_id)
            (run_path / "snapshots").mkdir(parents=True, exist_ok=True)
            (run_path / "payloads").mkdir(parents=True, exist_ok=True)
            _write_json(run_path / "manifest.json", manifest.to_dict())
            (run_path / "events.jsonl").touch(exist_ok=True)
            return manifest

    def _load_manifest(self, run_id: str) -> JadeStateManifest:
        return JadeStateManifest.from_dict(_read_json(self._run_path(run_id) / "manifest.json"))

    def _save_manifest(self, manifest: JadeStateManifest) -> None:
        _write_json(self._run_path(manifest.run_id) / "manifest.json", manifest.to_dict())

    def append_event(self, run_id: str, event: JadeStateEvent | dict[str, Any]) -> JadeStateEvent:
        with self._lock:
            state_event = event if isinstance(event, JadeStateEvent) else JadeStateEvent.from_dict(event)
            state_event.run_id = state_event.run_id or run_id
            run_path = self._run_path(run_id)
            if not run_path.exists():
                self.create_run(JadeStateManifest(run_id=run_id))
            _append_jsonl(run_path / "events.jsonl", state_event.to_dict())
            manifest = self._load_manifest(run_id)
            manifest.touch()
            self._save_manifest(manifest)
            return state_event

    def save_snapshot(self, run_id: str, snapshot: AgentRuntimeSnapshot) -> AgentRuntimeSnapshot:
        with self._lock:
            run_path = self._run_path(run_id)
            if not run_path.exists():
                self.create_run(JadeStateManifest(run_id=run_id))
            _write_json(run_path / "snapshots" / f"{snapshot.snapshot_id}.json", snapshot.to_dict())
            manifest = self._load_manifest(run_id)
            manifest.latest_snapshot_id = snapshot.snapshot_id
            manifest.touch()
            self._save_manifest(manifest)
            return snapshot

    def latest_snapshot(self, run_id: str) -> AgentRuntimeSnapshot | None:
        with self._lock:
            run_path = self._run_path(run_id)
            if not run_path.exists():
                return None
            manifest = self._load_manifest(run_id)
            snapshot_path: Path | None = None
            if manifest.latest_snapshot_id:
                candidate = run_path / "snapshots" / f"{manifest.latest_snapshot_id}.json"
                if candidate.exists():
                    snapshot_path = candidate
            if snapshot_path is None:
                snapshot_paths = sorted((run_path / "snapshots").glob("*.json"))
                if not snapshot_paths:
                    return None
                snapshot_path = snapshot_paths[-1]
            return AgentRuntimeSnapshot.from_dict(_read_json(snapshot_path))

    def load_run(self, run_id: str) -> JadeExecutionCapsule:
        with self._lock:
            return JadeExecutionCapsule.from_directory(self._run_path(run_id))

    def list_events(self, run_id: str, limit: int = 100) -> list[JadeStateEvent]:
        with self._lock:
            rows = _read_jsonl(self._run_path(run_id) / "events.jsonl")
            return [JadeStateEvent.from_dict(row) for row in rows[-limit:]]

    def inspect(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            return self.load_run(run_id).inspect()

    def export_run(self, run_id: str, destination: str | Path) -> Path:
        """Write a copy of a stored run to another .jgx directory."""

        capsule = self.load_run(run_id)
        return capsule.to_directory(destination)

    def list_runs(self) -> list[str]:
        return [
            path.name[:-4]
            for path in sorted(self.root.glob("*.jgx"))
            if path.is_dir()
        ]
