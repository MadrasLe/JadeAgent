"""Memory systems for persistent and shared agent context."""

from .base import BaseMemory
from .buffer import BufferMemory
from .router import InMemorySharedMemoryStore, MemoryRouter, MemoryStore, RedisMemoryStore

__all__ = [
    "BaseMemory",
    "BufferMemory",
    "MemoryStore",
    "InMemorySharedMemoryStore",
    "RedisMemoryStore",
    "MemoryRouter",
    "ShoreStoneMemory",
]


def ShoreStoneMemory(*args, **kwargs):
    """Lazy import for optional vector memory dependencies."""
    from .shorestone import ShoreStoneMemory as _SSM

    return _SSM(*args, **kwargs)
