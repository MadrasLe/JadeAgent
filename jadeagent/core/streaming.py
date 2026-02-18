"""
Token streaming protocol for JadeAgent.

Provides a unified streaming interface that works across
all backends (MegaGemm local, OpenAI, Groq, etc.).
"""

from __future__ import annotations

import sys
from typing import Iterator

from .types import Message, StreamChunk, ToolSchema
from ..backends.base import LLMBackend


def stream_tokens(
    backend: LLMBackend,
    messages: list[Message],
    tools: list[ToolSchema] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    print_live: bool = False,
) -> Iterator[StreamChunk]:
    """
    Stream tokens from any backend.

    Args:
        backend: LLM backend to stream from.
        messages: Conversation messages.
        tools: Optional tool schemas.
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.
        print_live: If True, print tokens to stdout in real-time.

    Yields:
        StreamChunk objects with token text and status.
    """
    for chunk in backend.stream(
        messages, tools=tools, temperature=temperature, max_tokens=max_tokens
    ):
        if print_live and chunk.token:
            print(chunk.token, end="", flush=True)

        yield chunk

    if print_live:
        print()  # Newline after streaming


def collect_stream(stream: Iterator[StreamChunk]) -> str:
    """Collect all tokens from a stream into a single string."""
    tokens = []
    for chunk in stream:
        if chunk.token:
            tokens.append(chunk.token)
    return "".join(tokens)
