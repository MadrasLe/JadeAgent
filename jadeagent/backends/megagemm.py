"""
MegaGemm backend — local GPU inference with KV cache persistence.

This is the ★ star backend of JadeAgent. It provides:
- Zero network latency (inference on local GPU)
- KV cache persistence across turns (skip re-prefill)
- KV cache CPU offloading for long agent contexts
- INT8/AWQ quantization support
- No API costs
"""

from __future__ import annotations

import json
import logging
import re
from typing import Iterator

from .base import LLMBackend
from ..core.types import (
    Message, Response, StreamChunk, ToolCall, ToolSchema, Usage,
)

logger = logging.getLogger("jadeagent.backends.megagemm")


# ----- Tool call parsing for local models ----------------------------------
# Different models use different formats for tool calls.
# We support Qwen3, LLaMA 3.1+, and generic JSON extraction.

_TOOL_CALL_PATTERNS = [
    # Qwen3 format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    re.compile(
        r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
        re.DOTALL,
    ),
    # Generic JSON with "name" and "arguments" keys
    re.compile(
        r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
        re.DOTALL,
    ),
]


def _parse_tool_calls_from_text(text: str) -> list[ToolCall] | None:
    """
    Extract tool calls from raw LLM output text.
    Handles multiple model formats (Qwen3, LLaMA, generic).
    """
    if not text:
        return None

    tool_calls = []
    call_id = 0

    # Try Qwen3 format first
    for match in _TOOL_CALL_PATTERNS[0].finditer(text):
        try:
            data = json.loads(match.group(1))
            tool_calls.append(ToolCall(
                id=f"call_{call_id}",
                name=data.get("name", ""),
                arguments=data.get("arguments", {}),
            ))
            call_id += 1
        except json.JSONDecodeError:
            continue

    if tool_calls:
        return tool_calls

    # Try generic JSON extraction
    # Look for JSON objects that have "name" and "arguments"
    try:
        # Find all JSON-like blocks
        json_blocks = re.findall(r'\{[^{}]*\{[^{}]*\}[^{}]*\}', text)
        for block in json_blocks:
            try:
                data = json.loads(block)
                if "name" in data and "arguments" in data:
                    tool_calls.append(ToolCall(
                        id=f"call_{call_id}",
                        name=data["name"],
                        arguments=data["arguments"],
                    ))
                    call_id += 1
            except json.JSONDecodeError:
                continue
    except Exception:
        pass

    return tool_calls if tool_calls else None


def _format_tools_for_prompt(tools: list[ToolSchema]) -> str:
    """Format tool schemas into a system prompt section."""
    if not tools:
        return ""

    tool_descriptions = []
    for t in tools:
        params_str = json.dumps(t.parameters, indent=2)
        tool_descriptions.append(
            f"### {t.name}\n"
            f"{t.description}\n"
            f"Parameters:\n```json\n{params_str}\n```"
        )

    return (
        "\n\n## Available Tools\n"
        "You can call tools by responding with a JSON object in this format:\n"
        '```json\n{"name": "tool_name", "arguments": {"param1": "value1"}}\n```\n'
        "When you want to call a tool, wrap it in <tool_call> tags:\n"
        '<tool_call>{"name": "tool_name", "arguments": {"param1": "value1"}}</tool_call>\n\n'
        + "\n\n".join(tool_descriptions)
    )


class MegaGemmBackend(LLMBackend):
    """
    Local GPU inference via MegaGemm engine.

    Example:
        backend = MegaGemmBackend(
            "Qwen/Qwen2.5-7B-Instruct",
            quantize="int8",
            kv_offload=True,
            num_cpu_blocks=2048,
        )
    """

    def __init__(
        self,
        model: str,
        quantize: str | None = None,
        kv_offload: bool = False,
        num_blocks: int = 4096,
        num_cpu_blocks: int = 0,
        gpu_window: int = 64,
        n_gpu_layers: int | None = None,
        dtype=None,
        **engine_kwargs,
    ):
        try:
            from megagemm.engine import InferenceEngine
        except ImportError:
            raise ImportError(
                "MegaGemm not found. Install it: pip install -e /path/to/MGRrmsnorm\n"
                "Or use OpenAICompatBackend for API-based inference."
            )

        self._model_name = model
        self._quantize = quantize

        # Build engine kwargs
        engine_kw = {
            "model_name": model,
            "num_blocks": num_blocks,
            **engine_kwargs,
        }
        if quantize:
            engine_kw["quantize"] = quantize
        if dtype:
            engine_kw["dtype"] = dtype
        if n_gpu_layers is not None:
            engine_kw["n_gpu_layers"] = n_gpu_layers
        if kv_offload:
            engine_kw["kv_offload"] = True
            engine_kw["num_cpu_blocks"] = num_cpu_blocks
            engine_kw["gpu_window"] = gpu_window

        logger.info(f"Initializing MegaGemm: {model} (quantize={quantize})")
        self.engine = InferenceEngine(**engine_kw)
        logger.info("MegaGemm engine ready.")

    def _messages_to_prompt(
        self, messages: list[Message], tools: list[ToolSchema] | None = None
    ) -> str:
        """Convert messages to a formatted prompt string."""
        # Try to use MegaGemm's built-in chat template
        chat_messages = []
        for msg in messages:
            role = str(msg.role)
            content = msg.content or ""

            # Inject tool schemas into system prompt
            if role == "system" and tools:
                content += _format_tools_for_prompt(tools)

            # Tool results → format as user message
            if role == "tool":
                tool_name = msg.name or "tool"
                content = (
                    f"<tool_response>\n"
                    f"Tool: {tool_name}\n"
                    f"Result: {content}\n"
                    f"</tool_response>"
                )
                role = "user"

            chat_messages.append({"role": role, "content": content})

        # Add tool prompt to system if no system message exists
        if tools and not any(m["role"] == "system" for m in chat_messages):
            tool_prompt = (
                "You are a helpful assistant with access to tools."
                + _format_tools_for_prompt(tools)
            )
            chat_messages.insert(0, {"role": "system", "content": tool_prompt})

        # Use MegaGemm's chat template formatting
        try:
            prompt = self.engine.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            # Fallback: simple concatenation
            parts = []
            for m in chat_messages:
                role = m["role"]
                content = m["content"]
                parts.append(f"<|{role}|>\n{content}")
            parts.append("<|assistant|>\n")
            prompt = "\n".join(parts)

        return prompt

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> Response:
        """Generate response using local MegaGemm engine."""
        prompt = self._messages_to_prompt(messages, tools)

        # Generate
        output = self.engine.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        # Try to parse tool calls from the output
        tool_calls = None
        if tools:
            tool_calls = _parse_tool_calls_from_text(output)

        return Response(
            content=output,
            tool_calls=tool_calls,
            model=self._model_name,
            finish_reason="stop",
        )

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> Iterator[StreamChunk]:
        """
        Stream tokens from MegaGemm.

        Note: This is a simulated stream — MegaGemm generates the full
        response and we yield it token by token. True streaming requires
        modifications to MegaGemm's decode loop (future work).
        """
        # For now: generate full response, then yield character by character
        response = self.chat(messages, tools, temperature, max_tokens, stop)
        text = response.content or ""

        # Yield word by word for smoother streaming effect
        words = text.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            yield StreamChunk(
                token=token,
                finished=(i == len(words) - 1),
                tool_calls=response.tool_calls if i == len(words) - 1 else None,
            )

    @property
    def supports_kv_persistence(self) -> bool:
        return True

    @property
    def supports_tool_calling(self) -> bool:
        return True  # Via prompt-based tool calling

    @property
    def name(self) -> str:
        q = f"/{self._quantize}" if self._quantize else ""
        return f"MegaGemm({self._model_name}{q})"
