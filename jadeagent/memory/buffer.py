"""
Sliding window buffer memory.

Simple memory that keeps the last N messages.
Useful for lightweight agents that don't need persistent storage.
"""

from __future__ import annotations

from collections import deque

from .base import BaseMemory


class BufferMemory(BaseMemory):
    """
    In-memory sliding window of recent memories.

    Example:
        memory = BufferMemory(max_size=100)
        memory.memorize("User likes Python")
        memory.memorize("User is working on AI project")
        results = memory.remember("programming language preference")
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._buffer: deque[dict] = deque(maxlen=max_size)

    def remember(self, query: str, k: int = 5) -> list[str]:
        """
        Simple keyword-based search over buffer.

        For semantic search, use ShoreStoneMemory instead.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Score each memory by keyword overlap
        scored = []
        for entry in self._buffer:
            content = entry["content"]
            content_lower = content.lower()
            content_words = set(content_lower.split())

            # Score: number of matching words
            overlap = len(query_words & content_words)
            if overlap > 0 or query_lower in content_lower:
                score = overlap + (2 if query_lower in content_lower else 0)
                scored.append((score, content))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Return top-k, or all if fewer than k
        if not scored:
            # Return most recent memories as fallback
            return [e["content"] for e in list(self._buffer)[-k:]]

        return [content for _, content in scored[:k]]

    def memorize(self, content: str, metadata: dict | None = None):
        """Store a memory in the buffer."""
        self._buffer.append({
            "content": content,
            "metadata": metadata or {},
        })

    def clear(self):
        """Clear all memories."""
        self._buffer.clear()

    @property
    def size(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"<BufferMemory(size={self.size}/{self.max_size})>"
