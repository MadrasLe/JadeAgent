"""
Multi-turn conversation session with optional KV cache persistence.

For MegaGemm backend: reuses KV cache across turns (skips re-prefill).
For API backends: sends full message history each time (standard behavior).
"""

from __future__ import annotations

import logging
from typing import Iterator

from .types import Message, Response, StreamChunk, ToolCall, ToolSchema, Role
from ..backends.base import LLMBackend
from ..state.snapshot import SessionSnapshot

logger = logging.getLogger("jadeagent.core.session")


class Session:
    """
    Multi-turn conversation session.

    Manages message history and provides a clean chat() interface.
    When using MegaGemm backend, enables KV cache persistence
    across turns for faster inference.

    Example:
        session = Session(backend, system_prompt="You are a helpful assistant.")
        r1 = session.chat("What is gravity?")
        r2 = session.chat("Can you explain more?")  # Reuses context!
    """

    def __init__(
        self,
        backend: LLMBackend,
        system_prompt: str | None = None,
    ):
        self.backend = backend
        self.messages: list[Message] = []

        if system_prompt:
            self.messages.append(Message.system(system_prompt))

    def chat(
        self,
        user_input: str,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Response:
        """
        Send a message and get a response.

        Args:
            user_input: The user's message text.
            tools: Optional tool schemas for function calling.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Response with content and/or tool_calls.
        """
        # Add user message to history
        self.messages.append(Message.user(user_input))

        # Generate response
        response = self.backend.chat(
            self.messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Add assistant response to history
        self.messages.append(Message.assistant(
            content=response.content,
            tool_calls=response.tool_calls,
        ))

        return response

    def add_tool_result(self, tool_call_id: str, name: str, result: str):
        """Add a tool execution result to the conversation history."""
        self.messages.append(Message.tool_result(tool_call_id, name, result))

    def stream_chat(
        self,
        user_input: str,
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[StreamChunk]:
        """
        Stream a response token by token.

        Args:
            user_input: The user's message text.
            tools: Optional tool schemas.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.

        Yields:
            StreamChunk objects.
        """
        self.messages.append(Message.user(user_input))

        full_content = []
        for chunk in self.backend.stream(
            self.messages, tools=tools,
            temperature=temperature, max_tokens=max_tokens,
        ):
            if chunk.token:
                full_content.append(chunk.token)
            yield chunk

        # Add the complete response to history
        content = "".join(full_content)
        self.messages.append(Message.assistant(content=content))

    def reset(self):
        """Clear conversation history, keeping system prompt."""
        system_msgs = [m for m in self.messages if m.role == Role.SYSTEM]
        self.messages = system_msgs

    def snapshot(self, metadata: dict | None = None) -> SessionSnapshot:
        """Capture restorable conversation state for a .jgx capsule."""
        return SessionSnapshot.from_messages(
            self.messages,
            backend=self.backend.name,
            metadata=dict(metadata or {}),
        )

    def restore_snapshot(self, snapshot: SessionSnapshot | dict):
        """Restore this session's message history from a snapshot."""
        session_snapshot = (
            snapshot
            if isinstance(snapshot, SessionSnapshot)
            else SessionSnapshot.from_dict(snapshot)
        )
        self.messages = session_snapshot.restore_messages()

    @classmethod
    def restore(cls, backend: LLMBackend, snapshot: SessionSnapshot | dict) -> "Session":
        """Create a new Session from a saved conversation snapshot."""
        session = cls(backend)
        session.restore_snapshot(snapshot)
        return session

    def fork(self) -> Session:
        """Create a branch of this session (for tree-of-thought)."""
        forked = Session(self.backend)
        forked.messages = [
            Message(role=m.role, content=m.content,
                    tool_calls=m.tool_calls, tool_call_id=m.tool_call_id,
                    name=m.name)
            for m in self.messages
        ]
        return forked

    @property
    def turn_count(self) -> int:
        """Number of user messages in the session."""
        return sum(1 for m in self.messages if m.role == Role.USER)

    @property
    def history(self) -> list[dict]:
        """Get message history as list of dicts."""
        return [m.to_dict() for m in self.messages]

    def __repr__(self) -> str:
        return f"<Session(turns={self.turn_count}, msgs={len(self.messages)})>"
