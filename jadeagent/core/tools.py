"""
@tool decorator and ToolRegistry for JadeAgent.

This module supports runtime-enforced policy metadata so the LLM cannot bypass
strict node or task permissions.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable, get_type_hints

from .types import ToolCall, ToolSchema
from ..governance import NodeManifest, TaskPolicy, evaluate_tool_call

logger = logging.getLogger("jadeagent.core.tools")

_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(py_type: type) -> dict[str, Any]:
    origin = getattr(py_type, "__origin__", None)
    if origin is not None:
        args = getattr(py_type, "__args__", ())
        if origin is list:
            item_type = args[0] if args else str
            return {"type": "array", "items": _python_type_to_json_schema(item_type)}
        if origin is dict:
            return {"type": "object"}
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _python_type_to_json_schema(non_none[0])

    return {"type": _TYPE_MAP.get(py_type, "string")}


def _emit_audit_event(audit_sink: Any, payload: dict[str, Any]):
    if audit_sink is None:
        return
    record_fn = getattr(audit_sink, "record_event", None)
    if callable(record_fn):
        try:
            record_fn(payload)
        except Exception:
            logger.debug("Audit sink rejected event", exc_info=True)


class Tool:
    """A registered tool with executable metadata."""

    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        safe_mode: bool = False,
        effects: list[str] | tuple[str, ...] | None = None,
        resource_refs: list[Any] | tuple[Any, ...] | None = None,
        read_path_args: list[str] | tuple[str, ...] | None = None,
        write_path_args: list[str] | tuple[str, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or f"Tool: {self.name}"
        self.safe_mode = safe_mode
        self.effects = tuple(str(effect) for effect in (effects or ()))
        self.resource_refs = tuple(resource_refs or ())
        self.read_path_args = tuple(str(arg) for arg in (read_path_args or ()))
        self.write_path_args = tuple(str(arg) for arg in (write_path_args or ()))
        self.metadata = dict(metadata or {})
        self.schema = self._build_schema()

    def _build_schema(self) -> ToolSchema:
        sig = inspect.signature(self.func)
        hints = get_type_hints(self.func)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            py_type = hints.get(param_name, str)
            prop = _python_type_to_json_schema(py_type)
            prop_desc = self._extract_param_doc(param_name)
            if prop_desc:
                prop["description"] = prop_desc
            properties[param_name] = prop

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
        doc = self.func.__doc__
        if not doc:
            return None

        lines = doc.split("\n")
        in_args = False
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("args:"):
                in_args = True
                continue
            if in_args:
                if stripped.startswith(f"{param_name}"):
                    parts = stripped.split(":", 1)
                    if len(parts) > 1:
                        return parts[1].strip()
                    parts = stripped.split(")", 1)
                    if len(parts) > 1:
                        return parts[1].strip().lstrip(":").strip()
                elif stripped and not stripped.startswith(" "):
                    in_args = False
        return None

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        node_manifest: NodeManifest | None = None,
        task_policy: TaskPolicy | None = None,
        cwd: str | None = None,
        audit_sink: Any = None,
        execution_context: dict[str, Any] | None = None,
    ) -> str:
        decision = evaluate_tool_call(
            self,
            arguments,
            node_manifest=node_manifest,
            task_policy=task_policy,
            cwd=cwd,
        )
        context = dict(execution_context or {})
        if not decision.allowed:
            _emit_audit_event(audit_sink, {
                "event_type": "policy_denied",
                "tool_name": self.name,
                "message": decision.reason,
                "resource": decision.resource,
                "action": decision.action,
                "scope": decision.scope,
                **context,
            })
            return f"❌ Policy denied: {decision.reason}"

        if self.safe_mode:
            print("\n[TOOL EXECUTION REQUEST]")
            print(f"Tool: {self.name}")
            print(f"Arguments: {json.dumps(arguments, indent=2)}")
            confirm = input(">> Allow? (y/n): ").strip().lower()
            if confirm not in ("y", "s", "yes", "sim"):
                return "❌ Action denied by user."

        try:
            result = self.func(**arguments)
            _emit_audit_event(audit_sink, {
                "event_type": "tool_called",
                "tool_name": self.name,
                "message": "tool executed",
                **context,
            })
            return str(result)
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {e}")
            _emit_audit_event(audit_sink, {
                "event_type": "tool_called",
                "tool_name": self.name,
                "message": f"tool error: {e}",
                **context,
            })
            return f"❌ Tool error: {e}"

    def __repr__(self) -> str:
        return f"<Tool({self.name})>"


def tool(
    description: str | None = None,
    name: str | None = None,
    safe_mode: bool = False,
    effects: list[str] | tuple[str, ...] | None = None,
    resource_refs: list[Any] | tuple[Any, ...] | None = None,
    read_path_args: list[str] | tuple[str, ...] | None = None,
    write_path_args: list[str] | tuple[str, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable:
    """
    Decorator to register a function as an agent tool.
    """

    def decorator(func: Callable) -> Tool:
        return Tool(
            func,
            name=name,
            description=description,
            safe_mode=safe_mode,
            effects=effects,
            resource_refs=resource_refs,
            read_path_args=read_path_args,
            write_path_args=write_path_args,
            metadata=metadata,
        )

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
        return [t.schema for t in self._tools.values()]

    def execute(
        self,
        tool_call: ToolCall,
        *,
        node_manifest: NodeManifest | None = None,
        task_policy: TaskPolicy | None = None,
        cwd: str | None = None,
        audit_sink: Any = None,
        execution_context: dict[str, Any] | None = None,
    ) -> str:
        tool_obj = self._tools.get(tool_call.name)
        if tool_obj is None:
            logger.warning(f"Tool not found: {tool_call.name}")
            return f"❌ Tool '{tool_call.name}' not found. Available: {list(self._tools.keys())}"
        return tool_obj.execute(
            tool_call.arguments,
            node_manifest=node_manifest,
            task_policy=task_policy,
            cwd=cwd,
            audit_sink=audit_sink,
            execution_context=execution_context,
        )

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"<ToolRegistry({self.names})>"
