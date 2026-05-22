"""
Async transport primitives for mesh communications.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any, Protocol

from .protocol import MeshEnvelope
from .transport import MeshTransport


class AsyncMeshTransport(Protocol):
    """Async transport interface used by AsyncMeshNode."""

    async def register(self, node: Any):
        """Register a local node handle on this transport."""
        ...

    async def unregister(self, node_id: str):
        """Unregister a node from this transport."""
        ...

    async def send(self, envelope: MeshEnvelope) -> int:
        """Send one envelope. Returns number of accepted deliveries."""
        ...

    async def recv(
        self,
        node_id: str,
        max_messages: int = 32,
        timeout: float | None = None,
    ) -> list[MeshEnvelope]:
        """Receive one or more inbound envelopes for a node."""
        ...


class AsyncMeshTransportAdapter:
    """
    Adapter that exposes a sync MeshTransport through the async interface.

    This keeps compatibility with existing transports while the native async
    transport path is adopted incrementally.
    """

    def __init__(self, transport: MeshTransport):
        self.transport = transport

    async def register(self, node: Any):
        await asyncio.to_thread(self.transport.register, node)

    async def unregister(self, node_id: str):
        await asyncio.to_thread(self.transport.unregister, node_id)

    async def send(self, envelope: MeshEnvelope) -> int:
        return int(await asyncio.to_thread(self.transport.send, envelope))

    async def recv(
        self,
        node_id: str,
        max_messages: int = 32,
        timeout: float | None = None,
    ) -> list[MeshEnvelope]:
        del timeout
        return await asyncio.to_thread(self.transport.poll, node_id, max_messages)


class AsyncInMemoryMeshBus:
    """
    In-process async transport for mesh envelopes.

    Unlike the sync test bus, this one is event-driven and allows nodes to
    await inbound traffic without polling loops.
    """

    def __init__(self):
        self._nodes: dict[str, Any] = {}
        self._queues: dict[str, asyncio.Queue[MeshEnvelope]] = {}

    async def register(self, node: Any):
        node_id = getattr(node, "node_id", None) or str(node)
        self._nodes[node_id] = node
        self._queues.setdefault(node_id, asyncio.Queue())

    async def unregister(self, node_id: str):
        self._nodes.pop(node_id, None)
        self._queues.pop(node_id, None)

    async def send(self, envelope: MeshEnvelope) -> int:
        delivered = 0
        if envelope.destination is None:
            for node_id, queue in self._queues.items():
                if node_id == envelope.source:
                    continue
                await queue.put(copy.deepcopy(envelope))
                delivered += 1
            return delivered

        queue = self._queues.get(envelope.destination)
        if queue is None:
            return 0
        await queue.put(copy.deepcopy(envelope))
        return 1

    async def recv(
        self,
        node_id: str,
        max_messages: int = 32,
        timeout: float | None = None,
    ) -> list[MeshEnvelope]:
        queue = self._queues.setdefault(node_id, asyncio.Queue())
        if max_messages <= 0:
            return []

        try:
            if timeout is None:
                first = await queue.get()
            elif timeout <= 0:
                first = queue.get_nowait()
            else:
                first = await asyncio.wait_for(queue.get(), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            return []

        envelopes = [first]
        while len(envelopes) < max_messages:
            try:
                envelopes.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return envelopes

    async def run_until_idle(self, max_cycles: int = 2000) -> int:
        cycles = 0
        for _ in range(max_cycles):
            progressed = False
            for node in list(self._nodes.values()):
                step = getattr(node, "astep", None)
                if callable(step):
                    progressed = bool(await step(timeout=0.0)) or progressed
            cycles += 1
            if not progressed:
                pending = False
                for queue in self._queues.values():
                    if not queue.empty():
                        pending = True
                        break
                if not pending:
                    break
        return cycles
