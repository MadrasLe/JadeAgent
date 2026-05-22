"""
OpenAI-compatible backend — works with ANY provider.

Supports: OpenAI, Groq, Anthropic (via proxy), Together, Fireworks,
Ollama, LM Studio, or any server exposing /v1/chat/completions.
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Iterator

from .base import LLMBackend
from ..core.types import (
    Message, Response, StreamChunk, ToolCall, ToolSchema, Usage,
)

logger = logging.getLogger("jadeagent.backends.openai_compat")


class KeyRotator:
    """Rotate through API keys on rate-limit errors (from JadeHeavy)."""

    def __init__(self, keys: list[str]):
        self.keys = keys
        self.index = 0

    @property
    def current(self) -> str:
        return self.keys[self.index]

    def rotate(self) -> str:
        self.index = (self.index + 1) % len(self.keys)
        logger.warning(f"Rotating to API key #{self.index + 1}/{len(self.keys)}")
        return self.current


class OpenAICompatBackend(LLMBackend):
    """
    Universal backend for any OpenAI-compatible API.

    Examples:
        # OpenAI
        backend = OpenAICompatBackend("gpt-4o", api_key="sk-...")

        # Groq
        backend = OpenAICompatBackend(
            "llama-3.1-8b-instant",
            base_url="https://api.groq.com/openai/v1",
            api_key="gsk_...",
        )

        # Ollama (local)
        backend = OpenAICompatBackend(
            "qwen3:8b",
            base_url="http://localhost:11434/v1",
        )

        # Multiple keys with rotation
        backend = OpenAICompatBackend(
            "moonshotai/kimi-k2-instruct-0905",
            base_url="https://api.groq.com/openai/v1",
            api_keys=["gsk_key1", "gsk_key2", "gsk_key3"],
        )
    """

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_keys: list[str] | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Key management
        keys = api_keys or ([api_key] if api_key else [""])
        self._rotator = KeyRotator(keys)

        # Lazy-init openai client
        self._client = None

    def _get_client(self):
        """Lazy-initialize OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "Install openai: pip install openai"
                )
            self._client = OpenAI(
                api_key=self._rotator.current,
                base_url=self.base_url,
            )
        return self._client

    def _rebuild_client(self):
        """Rebuild client with rotated key."""
        key = self._rotator.rotate()
        from openai import OpenAI
        self._client = OpenAI(api_key=key, base_url=self.base_url)

    def _parse_tool_calls(self, raw_tool_calls) -> list[ToolCall]:
        """Parse OpenAI tool_calls response into our ToolCall objects."""
        result = []
        for tc in raw_tool_calls:
            func = tc.function
            try:
                args = json.loads(func.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {"raw": func.arguments}
            result.append(ToolCall(
                id=tc.id,
                name=func.name,
                arguments=args,
            ))
        return result

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> Response:
        """Send messages to the API and return a complete response."""
        client = self._get_client()

        # Build request kwargs
        kwargs: dict = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            kwargs["stop"] = stop
        if tools:
            kwargs["tools"] = [t.to_dict() for t in tools]

        # Retry loop with key rotation
        last_error = None
        for attempt in range(self.max_retries):
            try:
                completion = client.chat.completions.create(**kwargs)
                choice = completion.choices[0]

                # Parse tool calls if present
                tool_calls = None
                if choice.message.tool_calls:
                    tool_calls = self._parse_tool_calls(choice.message.tool_calls)

                # Parse usage
                usage = None
                if completion.usage:
                    usage = Usage(
                        prompt_tokens=completion.usage.prompt_tokens,
                        completion_tokens=completion.usage.completion_tokens,
                        total_tokens=completion.usage.total_tokens,
                    )

                return Response(
                    content=choice.message.content,
                    tool_calls=tool_calls,
                    usage=usage,
                    model=completion.model,
                    finish_reason=choice.finish_reason,
                )

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Rate limit → rotate key and retry
                if "rate" in error_str or "429" in error_str:
                    logger.warning(f"Rate limit hit, rotating key (attempt {attempt+1})")
                    self._rebuild_client()
                    client = self._client
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue

                # Server error → retry with backoff
                if "500" in error_str or "503" in error_str:
                    logger.warning(f"Server error, retrying (attempt {attempt+1})")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue

                # Other error → raise immediately
                raise

        raise RuntimeError(
            f"Failed after {self.max_retries} retries. Last error: {last_error}"
        )

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> Iterator[StreamChunk]:
        """Stream tokens from the API."""
        client = self._get_client()

        kwargs: dict = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if stop:
            kwargs["stop"] = stop
        if tools:
            kwargs["tools"] = [t.to_dict() for t in tools]

        stream = client.chat.completions.create(**kwargs)

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            finished = chunk.choices[0].finish_reason is not None

            token = delta.content or ""
            yield StreamChunk(token=token, finished=finished)

    @property
    def name(self) -> str:
        provider = self.base_url.split("//")[-1].split("/")[0].split(".")[0]
        return f"OpenAICompat({provider}/{self.model})"

    def __repr__(self) -> str:
        return f"<{self.name}>"
