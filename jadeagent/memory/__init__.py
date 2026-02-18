"""Memory systems for persistent agent context."""

from .base import BaseMemory
from .buffer import BufferMemory

__all__ = ["BaseMemory", "BufferMemory"]

# Lazy import for ShoreStone (requires chromadb + sentence-transformers)
def ShoreStoneMemory(*args, **kwargs):
    from .shorestone import ShoreStoneMemory as _SSM
    return _SSM(*args, **kwargs)
