"""JadeAgent - next-gen agent runtime with executable governance."""

from .a2a import a2a_request_to_task, manifest_to_agent_card, task_to_a2a_request
from .core.agent import Agent
from .core.session import Session
from .core.tools import tool
from .core.types import Message, Response, StreamChunk, ToolCall
from .governance import (
    AccessGrant,
    FilesystemPolicy,
    MemoryMount,
    NodeManifest,
    PolicyBundle,
    TaskPolicy,
)
from .state import (
    AgentRuntimeSnapshot,
    FileStateStore,
    InMemoryStateStore,
    JadeExecutionCapsule,
    JadeStateEvent,
    JadeStateManifest,
    SessionSnapshot,
    SqliteStateStore,
    event_chain_hash,
    inspect_jgx,
    load_jgx,
    redact_secrets,
    verify_capsule,
    write_jgx,
)

__all__ = [
    "Agent",
    "tool",
    "Message",
    "Response",
    "ToolCall",
    "StreamChunk",
    "Session",
    "manifest_to_agent_card",
    "task_to_a2a_request",
    "a2a_request_to_task",
    "AccessGrant",
    "FilesystemPolicy",
    "MemoryMount",
    "NodeManifest",
    "PolicyBundle",
    "TaskPolicy",
    "AgentRuntimeSnapshot",
    "FileStateStore",
    "InMemoryStateStore",
    "JadeExecutionCapsule",
    "JadeStateEvent",
    "JadeStateManifest",
    "SessionSnapshot",
    "SqliteStateStore",
    "event_chain_hash",
    "inspect_jgx",
    "load_jgx",
    "redact_secrets",
    "verify_capsule",
    "write_jgx",
]

__version__ = "0.1.0"
