"""Abstract base class for all LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from ..core.types import Message, Response, StreamChunk, ToolSchema


class LLMBackend(ABC):
    """
    Unified interface for LLM inference.

    All backends (MegaGemm local, OpenAI, Groq, Ollama, etc.)
    implement this interface, making agents backend-agnostic.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> Response:
        """
        Generate a complete response from a list of messages.

        Args:
            messages: Conversation history as Message objects.
            tools: Optional tool schemas for function calling.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens to generate.
            stop: Optional stop sequences.

        Returns:
            Response with content and/or tool_calls.
        """
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> Iterator[StreamChunk]:
        """
        Stream tokens one by one.

        Yields:
            StreamChunk with token text and finished flag.
        """
        ...

    @property
    def supports_kv_persistence(self) -> bool:
        """Whether this backend can persist KV cache across turns.
        Only MegaGemm returns True."""
        return False

    @property
    def supports_tool_calling(self) -> bool:
        """Whether this backend supports native tool/function calling."""
        return True

    @property
    def name(self) -> str:
        """Human-readable backend name."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"<{self.name}>"
