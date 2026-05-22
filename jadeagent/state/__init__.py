"""Governed execution state for JadeAgent.

The .jgx layer treats an agent run as a portable state machine: events describe
what happened, snapshots describe where execution can resume, and manifests
record the compatibility gates required before restore.
"""

from .artifact import (
    JGX_MAGIC,
    JadeExecutionCapsule,
    inspect_jgx,
    load_jgx,
    write_jgx,
)
from .compatibility import CompatibilityReport, validate_restore_compatibility
from .events import JadeStateEvent
from .integrity import event_chain_hash, find_secret_paths, redact_secrets, snapshot_hashes, verify_capsule
from .manifest import JadeStateManifest, canonical_json_hash, fingerprint_mapping
from .snapshot import (
    AgentRuntimeSnapshot,
    GraphRuntimeSnapshot,
    MeshRuntimeSnapshot,
    SessionSnapshot,
)
from .sqlite_store import SqliteStateStore
from .store import FileStateStore, InMemoryStateStore, StateStore

__all__ = [
    "JGX_MAGIC",
    "AgentRuntimeSnapshot",
    "CompatibilityReport",
    "FileStateStore",
    "GraphRuntimeSnapshot",
    "InMemoryStateStore",
    "JadeExecutionCapsule",
    "JadeStateEvent",
    "JadeStateManifest",
    "MeshRuntimeSnapshot",
    "SessionSnapshot",
    "SqliteStateStore",
    "StateStore",
    "canonical_json_hash",
    "event_chain_hash",
    "find_secret_paths",
    "fingerprint_mapping",
    "inspect_jgx",
    "load_jgx",
    "redact_secrets",
    "snapshot_hashes",
    "validate_restore_compatibility",
    "verify_capsule",
    "write_jgx",
]
