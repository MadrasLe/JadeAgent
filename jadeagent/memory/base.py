"""
Base class for memory systems.

Memory allows agents to persist information across conversations
and retrieve relevant context based on semantic similarity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseMemory(ABC):
    """Abstract base for all memory systems."""

    @abstractmethod
    def remember(self, query: str, k: int = 5) -> list[str]:
        """
        Retrieve relevant memories for a given query.

        Args:
            query: Search query (semantic similarity).
            k: Number of memories to retrieve.

        Returns:
            List of relevant memory strings.
        """
        ...

    @abstractmethod
    def memorize(self, content: str, metadata: dict | None = None):
        """
        Store a new memory.

        Args:
            content: The text content to memorize.
            metadata: Optional metadata (source, timestamp, etc).
        """
        ...

    @abstractmethod
    def clear(self):
        """Clear all memories."""
        ...

    @property
    def size(self) -> int:
        """Number of stored memories."""
        return 0
