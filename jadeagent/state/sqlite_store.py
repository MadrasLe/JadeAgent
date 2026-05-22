"""SQLite-backed StateStore for Jade governed execution capsules."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .artifact import JadeExecutionCapsule
from .events import JadeStateEvent
from .manifest import JadeStateManifest
from .snapshot import AgentRuntimeSnapshot
from .store import StateStore


class SqliteStateStore(StateStore):
    """Durable local SQLite store for JGX manifests, events, and snapshots."""

    def __init__(self, path: str | Path = ".jade_state.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS manifests (
                    run_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    data TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    phase TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    data TEXT NOT NULL,
                    UNIQUE(run_id, snapshot_id)
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_run_seq ON events(run_id, sequence)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_run_seq ON snapshots(run_id, sequence)")
            self._conn.commit()

    def _dump(self, data: dict[str, Any]) -> str:
        return json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":"))

    def _load_manifest(self, run_id: str) -> JadeStateManifest | None:
        row = self._conn.execute(
            "SELECT data FROM manifests WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return JadeStateManifest.from_dict(json.loads(row["data"]))

    def _save_manifest(self, manifest: JadeStateManifest) -> None:
        self._conn.execute(
            """
            INSERT INTO manifests(run_id, data, updated_at)
            VALUES(?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                data = excluded.data,
                updated_at = excluded.updated_at
            """,
            (manifest.run_id, self._dump(manifest.to_dict()), manifest.updated_at),
        )

    def _ensure_run(self, run_id: str) -> JadeStateManifest:
        manifest = self._load_manifest(run_id)
        if manifest is None:
            manifest = JadeStateManifest(run_id=run_id)
            self._save_manifest(manifest)
        return manifest

    def create_run(self, manifest: JadeStateManifest) -> JadeStateManifest:
        with self._lock:
            manifest.touch()
            self._save_manifest(manifest)
            self._conn.commit()
            return manifest

    def append_event(self, run_id: str, event: JadeStateEvent | dict[str, Any]) -> JadeStateEvent:
        with self._lock:
            state_event = event if isinstance(event, JadeStateEvent) else JadeStateEvent.from_dict(event)
            state_event.run_id = state_event.run_id or run_id
            manifest = self._ensure_run(run_id)
            self._conn.execute(
                "INSERT INTO events(run_id, event_id, timestamp, data) VALUES(?, ?, ?, ?)",
                (
                    run_id,
                    state_event.event_id,
                    state_event.timestamp,
                    self._dump(state_event.to_dict()),
                ),
            )
            manifest.touch()
            self._save_manifest(manifest)
            self._conn.commit()
            return state_event

    def save_snapshot(self, run_id: str, snapshot: AgentRuntimeSnapshot) -> AgentRuntimeSnapshot:
        with self._lock:
            manifest = self._ensure_run(run_id)
            self._conn.execute(
                """
                INSERT INTO snapshots(run_id, snapshot_id, created_at, phase, step, data)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, snapshot_id) DO UPDATE SET
                    created_at = excluded.created_at,
                    phase = excluded.phase,
                    step = excluded.step,
                    data = excluded.data
                """,
                (
                    run_id,
                    snapshot.snapshot_id,
                    snapshot.created_at,
                    snapshot.phase,
                    snapshot.step,
                    self._dump(snapshot.to_dict()),
                ),
            )
            manifest.latest_snapshot_id = snapshot.snapshot_id
            manifest.touch()
            self._save_manifest(manifest)
            self._conn.commit()
            return snapshot

    def latest_snapshot(self, run_id: str) -> AgentRuntimeSnapshot | None:
        with self._lock:
            manifest = self._load_manifest(run_id)
            if manifest is None:
                return None
            row = None
            if manifest.latest_snapshot_id:
                row = self._conn.execute(
                    "SELECT data FROM snapshots WHERE run_id = ? AND snapshot_id = ?",
                    (run_id, manifest.latest_snapshot_id),
                ).fetchone()
            if row is None:
                row = self._conn.execute(
                    "SELECT data FROM snapshots WHERE run_id = ? ORDER BY sequence DESC LIMIT 1",
                    (run_id,),
                ).fetchone()
            if row is None:
                return None
            return AgentRuntimeSnapshot.from_dict(json.loads(row["data"]))

    def load_run(self, run_id: str) -> JadeExecutionCapsule:
        with self._lock:
            manifest = self._load_manifest(run_id)
            if manifest is None:
                raise KeyError(f"run not found: {run_id}")
            event_rows = self._conn.execute(
                "SELECT data FROM events WHERE run_id = ? ORDER BY sequence ASC",
                (run_id,),
            ).fetchall()
            snapshot_rows = self._conn.execute(
                "SELECT data FROM snapshots WHERE run_id = ? ORDER BY sequence ASC",
                (run_id,),
            ).fetchall()
            return JadeExecutionCapsule(
                manifest=manifest,
                events=[JadeStateEvent.from_dict(json.loads(row["data"])) for row in event_rows],
                snapshots=[AgentRuntimeSnapshot.from_dict(json.loads(row["data"])) for row in snapshot_rows],
            )

    def list_events(self, run_id: str, limit: int = 100) -> list[JadeStateEvent]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT data FROM events
                WHERE run_id = ?
                ORDER BY sequence DESC
                LIMIT ?
                """,
                (run_id, max(int(limit), 1)),
            ).fetchall()
            return [
                JadeStateEvent.from_dict(json.loads(row["data"]))
                for row in reversed(rows)
            ]

    def inspect(self, run_id: str) -> dict[str, Any]:
        return self.load_run(run_id).inspect()

    def list_runs(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT run_id FROM manifests ORDER BY updated_at DESC").fetchall()
            return [str(row["run_id"]) for row in rows]

    def export_run(self, run_id: str, destination: str | Path) -> Path:
        capsule = self.load_run(run_id)
        return capsule.to_directory(destination)
