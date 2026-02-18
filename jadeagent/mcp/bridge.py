"""
Bridge MCP tools to JadeAgent's Tool system.

Converts MCP tool schemas to JadeAgent Tools so agents can
call MCP server tools seamlessly through the ReAct loop.
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
    A JadeAgent Tool that proxies calls to an MCP server.

    Created automatically by bridge_mcp_tools() — you don't need
    to instantiate this directly.
    """

    def __init__(self, mcp_client: MCPClient, tool_schema: dict):
        self._mcp_client = mcp_client
        self._mcp_schema = tool_schema

        # Extract schema info
        self._tool_name = tool_schema["name"]
        self._description = tool_schema.get("description", "")
        self._input_schema = tool_schema.get("inputSchema", {
            "type": "object",
            "properties": {},
        })

        # Build a dummy callable (we override execute())
        def _placeholder(**kwargs):
            pass

        super().__init__(
            func=_placeholder,
            name=self._tool_name,
            description=self._description,
        )

        # Override the schema with MCP's actual schema (after super sets it)
        self.schema = ToolSchema(
            name=self._tool_name,
            description=self._description,
            parameters=self._input_schema,
        )

    def __call__(self, **kwargs) -> str:
        """Call the MCP tool via the client."""
        logger.debug(f"MCP tool call: {self._tool_name}({kwargs})")
        return self._mcp_client.call_tool(self._tool_name, kwargs)

    def execute(self, arguments: dict) -> str:
        """Override Tool.execute to proxy to MCP server."""
        try:
            return self(**arguments)
        except Exception as e:
            logger.error(f"MCP tool {self._tool_name} failed: {e}")
            return f"Error calling MCP tool: {e}"


def bridge_mcp_tools(mcp_client: MCPClient) -> list[MCPTool]:
    """
    Convert all discovered MCP tools to JadeAgent Tools.

    Args:
        mcp_client: Connected MCPClient with discovered tools.

    Returns:
        List of MCPTool objects ready for Agent(tools=...).

    Example:
        mcp = MCPClient("npx @modelcontextprotocol/server-filesystem /tmp")
        tools = bridge_mcp_tools(mcp)
        agent = Agent(backend, tools=tools)
    """
    bridged = []
    for tool_schema in mcp_client.tool_schemas:
        try:
            mcp_tool = MCPTool(mcp_client, tool_schema)
            bridged.append(mcp_tool)
            logger.debug(f"Bridged MCP tool: {mcp_tool.name}")
        except Exception as e:
            logger.warning(f"Failed to bridge MCP tool '{tool_schema.get('name', '?')}': {e}")

    logger.info(f"Bridged {len(bridged)} MCP tools to JadeAgent")
    return bridged
