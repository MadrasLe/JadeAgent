"""
Transport interfaces for mesh communications.
"""

from __future__ import annotations

from typing import Any, Protocol

from .protocol import MeshEnvelope


class MeshTransport(Protocol):
    """Minimal transport interface used by MeshNode."""

    def register(self, node: Any):
        """Register a local node handle on this transport."""
        ...

    def unregister(self, node_id: str):
        """Unregister a node from this transport."""
        ...

    def send(self, envelope: MeshEnvelope) -> int:
        """Send one envelope. Returns number of accepted deliveries."""
        ...

    def poll(self, node_id: str, max_messages: int = 32) -> list[MeshEnvelope]:
        """Pull inbound envelopes for a node."""
        ...

