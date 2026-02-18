"""
@tool decorator and ToolRegistry for JadeAgent.

Automatically extracts JSON schemas from Python type hints,
compatible with OpenAI function calling format.

Evolved from CodeJade's ToolManager with improvements:
- Decorator-based registration
- Auto schema extraction from type hints
- Safe mode support (user confirmation for dangerous ops)
- Compatible with any backend (API or local)
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable, get_type_hints

from .types import ToolCall, ToolSchema

logger = logging.getLogger("jadeagent.core.tools")

# Python type → JSON Schema type mapping
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(py_type: type) -> dict[str, str]:
    """Convert a Python type hint to a JSON Schema type."""
    # Handle Optional (Union[X, None])
    origin = getattr(py_type, "__origin__", None)
    if origin is not None:
        args = getattr(py_type, "__args__", ())
        # Handle list[X]
        if origin is list:
            item_type = args[0] if args else str
            return {"type": "array", "items": _python_type_to_json_schema(item_type)}
        # Handle dict[str, X]
        if origin is dict:
            return {"type": "object"}
        # Handle Optional[X] (Union[X, None])
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _python_type_to_json_schema(non_none[0])

    return {"type": _TYPE_MAP.get(py_type, "string")}


class Tool:
    """A registered tool with its function and schema."""

    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        safe_mode: bool = False,
    ):
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or f"Tool: {self.name}"
        self.safe_mode = safe_mode
        self.schema = self._build_schema()

    def _build_schema(self) -> ToolSchema:
        """Build JSON schema from function signature and type hints."""
        sig = inspect.signature(self.func)
        hints = get_type_hints(self.func)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Get type hint
            py_type = hints.get(param_name, str)
            prop = _python_type_to_json_schema(py_type)

            # Get description from docstring if available
            # (simple parsing of Google-style docstrings)
            prop_desc = self._extract_param_doc(param_name)
            if prop_desc:
                prop["description"] = prop_desc

            properties[param_name] = prop

            # Required if no default value
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        parameters = {
            "type": "object",
            "properties": properties,
        }
        if required:
            parameters["required"] = required

        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=parameters,
        )

    def _extract_param_doc(self, param_name: str) -> str | None:
        """Extract parameter description from Google-style docstring."""
        doc = self.func.__doc__
        if not doc:
            return None

        # Look for "Args:" section
        lines = doc.split("\n")
        in_args = False
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("args:"):
                in_args = True
                continue
            if in_args:
                if stripped.startswith(f"{param_name}"):
                    # Extract description after colon
                    parts = stripped.split(":", 1)
                    if len(parts) > 1:
                        return parts[1].strip()
                    # Check if type annotation is included
                    parts = stripped.split(")", 1)
                    if len(parts) > 1:
                        return parts[1].strip().lstrip(":").strip()
                elif stripped and not stripped.startswith(" "):
                    in_args = False  # Exited Args section
        return None

    def execute(self, arguments: dict[str, Any]) -> str:
        """Execute the tool with given arguments."""
        if self.safe_mode:
            print(f"\n⚠️  [TOOL EXECUTION REQUEST]")
            print(f"Tool: {self.name}")
            print(f"Arguments: {json.dumps(arguments, indent=2)}")
            confirm = input(">> Allow? (y/n): ").strip().lower()
            if confirm not in ("y", "s", "yes", "sim"):
                return "❌ Action denied by user."

        try:
            result = self.func(**arguments)
            return str(result)
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {e}")
            return f"❌ Tool error: {e}"

    def __repr__(self) -> str:
        return f"<Tool({self.name})>"


def tool(
    description: str | None = None,
    name: str | None = None,
    safe_mode: bool = False,
) -> Callable:
    """
    Decorator to register a function as an agent tool.

    Usage:
        @tool(description="Search the web for information")
        def web_search(query: str) -> str:
            return requests.get(f"https://api.search.com?q={query}").text

        @tool(description="Execute shell command", safe_mode=True)
        def shell(command: str) -> str:
            return subprocess.run(command, shell=True, capture_output=True).stdout

    Args:
        description: Human-readable description of what the tool does.
        name: Override the function name as the tool name.
        safe_mode: If True, ask for user confirmation before execution.
    """
    def decorator(func: Callable) -> Tool:
        return Tool(func, name=name, description=description, safe_mode=safe_mode)

    # Allow @tool without arguments: @tool
    if callable(description):
        func = description
        return Tool(func)

    return decorator


class ToolRegistry:
    """Registry of tools available to an agent."""

    def __init__(self, tools: list[Tool | Callable] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in (tools or []):
            self.register(t)

    def register(self, t: Tool | Callable):
        """Register a tool. Accepts Tool objects or plain functions."""
        if isinstance(t, Tool):
            self._tools[t.name] = t
        elif callable(t):
            wrapped = Tool(t)
            self._tools[wrapped.name] = wrapped
        else:
            raise TypeError(f"Expected Tool or Callable, got {type(t)}")

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def schemas(self) -> list[ToolSchema]:
        """Get all tool schemas for passing to LLM."""
        return [t.schema for t in self._tools.values()]

    def execute(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return the result as string."""
        tool_obj = self._tools.get(tool_call.name)
        if tool_obj is None:
            logger.warning(f"Tool not found: {tool_call.name}")
            return f"❌ Tool '{tool_call.name}' not found. Available: {list(self._tools.keys())}"
        return tool_obj.execute(tool_call.arguments)

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"<ToolRegistry({self.names})>"
