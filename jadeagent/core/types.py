"""Core type definitions for JadeAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Message role in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """A single message in a conversation."""
    role: Role | str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool result messages
    name: str | None = None  # Tool name for tool results

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible message dict."""
        d: dict[str, Any] = {"role": str(self.role)}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": __import__("json").dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str | None = None,
                  tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool_result(cls, tool_call_id: str, name: str, content: str) -> Message:
        return cls(role=Role.TOOL, content=content,
                   tool_call_id=tool_call_id, name=name)


@dataclass
class ToolSchema:
    """JSON schema for a tool (OpenAI function calling format)."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class Usage:
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Response:
    """Complete LLM response."""
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: Usage | None = None
    model: str | None = None
    finish_reason: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class StreamChunk:
    """A single streaming chunk."""
    token: str = ""
    finished: bool = False
    tool_calls: list[ToolCall] | None = None  # Accumulated tool calls


@dataclass
class AgentEvent:
    """Event emitted during agent execution."""
    type: str  # "thinking", "tool_call", "tool_result", "answer", "error"
    content: str | None = None
    tool_call: ToolCall | None = None
    tool_result: str | None = None
    step: int = 0


@dataclass
class AgentResult:
    """Result of an agent run."""
    answer: str
    steps: int = 0
    tool_calls_made: list[ToolCall] = field(default_factory=list)
    events: list[AgentEvent] = field(default_factory=list)
    usage: Usage | None = None
