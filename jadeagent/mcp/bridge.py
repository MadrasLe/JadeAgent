"""
Bridge MCP tools to JadeAgent's Tool system.

Converts MCP tool schemas to JadeAgent tools so agents can call MCP server
tools through the normal runtime policy and audit pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from ..core.tools import Tool
from ..core.types import ToolSchema

if TYPE_CHECKING:
    from .client import MCPClient

logger = logging.getLogger("jadeagent.mcp.bridge")


class MCPTool(Tool):
    """
    A JadeAgent tool that proxies calls to an MCP server.

    Created automatically by :func:`bridge_mcp_tools`.
    """

    def __init__(self, mcp_client: MCPClient, tool_schema: dict):
        self._mcp_client = mcp_client
        self._mcp_schema = tool_schema
        self._tool_name = str(tool_schema["name"])
        self._description = str(tool_schema.get("description", ""))
        self._input_schema = tool_schema.get("inputSchema", {
            "type": "object",
            "properties": {},
        })

        def _proxy(**kwargs):
            logger.debug("MCP tool call: %s(%s)", self._tool_name, kwargs)
            return self._mcp_client.call_tool(self._tool_name, kwargs)

        super().__init__(
            func=_proxy,
            name=self._tool_name,
            description=self._description,
            resource_refs=[f"tool.execute:{self._tool_name}"],
            metadata={
                "mcp_tool": True,
                "mcp_schema_name": self._tool_name,
            },
        )
        self.schema = ToolSchema(
            name=self._tool_name,
            description=self._description,
            parameters=self._input_schema,
        )


def bridge_mcp_tools(mcp_client: MCPClient) -> list[MCPTool]:
    """
    Convert all discovered MCP tools to JadeAgent Tool objects.
    """
    bridged: list[MCPTool] = []
    for tool_schema in mcp_client.tool_schemas:
        try:
            bridged.append(MCPTool(mcp_client, tool_schema))
            logger.debug("Bridged MCP tool: %s", tool_schema.get("name"))
        except Exception as exc:
            logger.warning(
                "Failed to bridge MCP tool '%s': %s",
                tool_schema.get("name", "?"),
                exc,
            )
    logger.info("Bridged %s MCP tools to JadeAgent", len(bridged))
    return bridged
