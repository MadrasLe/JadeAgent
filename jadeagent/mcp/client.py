"""
MCP (Model Context Protocol) client for JadeAgent.

Connects to MCP servers via stdio transport and discovers/calls tools
using JSON-RPC 2.0. MCP tools are automatically bridged to JadeAgent's
ToolRegistry so agents can use them like native tools.

Based on MCP spec 2025-03-26:
- Transport: stdio (server launched as subprocess)
- Discovery: tools/list
- Execution: tools/call
- Protocol: JSON-RPC 2.0

Example:
    from jadeagent.mcp import MCPClient

    mcp = MCPClient("npx @modelcontextprotocol/server-filesystem /tmp")
    agent = Agent(backend, tools=mcp.tools)
    result = agent.run("List all files in /tmp")
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import uuid
from typing import Any

logger = logging.getLogger("jadeagent.mcp")


class MCPClient:
    """
    Connects to an MCP server via stdio transport.

    Launches the server as a subprocess, initializes the protocol,
    discovers tools, and provides methods to call them.

    Args:
        command: Shell command to launch the MCP server.
            E.g.: "npx @modelcontextprotocol/server-filesystem /tmp"
            Or: "python my_mcp_server.py"
        env: Optional environment variables for the subprocess.
        timeout: Timeout in seconds for JSON-RPC calls.

    Example:
        mcp = MCPClient("npx @modelcontextprotocol/server-filesystem /tmp")
        print(mcp.tool_names)  # ['read_file', 'write_file', 'list_dir', ...]

        # Get tools bridged to JadeAgent format
        tools = mcp.tools  # List of Tool objects

        # Use with an agent
        agent = Agent(backend, tools=tools)
        result = agent.run("Read the file /tmp/hello.txt")
    """

    def __init__(
        self,
        command: str,
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
    ):
        self.command = command
        self.timeout = timeout
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._msg_id = 0
        self._raw_tools: list[dict] = []
        self._initialized = False

        # Start the server
        self._start_server(env)

        # Initialize MCP protocol
        self._initialize()

        # Discover tools
        self._discover_tools()

    def _start_server(self, env: dict | None = None):
        """Launch the MCP server as a subprocess."""
        import shlex
        import os

        full_env = dict(os.environ)
        if env:
            full_env.update(env)

        # Parse command
        if sys.platform == "win32":
            args = self.command  # Windows handles string commands
            shell = True
        else:
            args = shlex.split(self.command)
            shell = False

        logger.info(f"Starting MCP server: {self.command}")

        self._process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=full_env,
            shell=shell,
            text=True,
            bufsize=0,
        )

    def _next_id(self) -> int:
        """Generate unique JSON-RPC message ID."""
        self._msg_id += 1
        return self._msg_id

    def _send_request(self, method: str, params: dict | None = None) -> Any:
        """
        Send a JSON-RPC 2.0 request and wait for response.

        Args:
            method: JSON-RPC method name.
            params: Optional parameters dict.

        Returns:
            The 'result' field from the response.

        Raises:
            RuntimeError: If the server returns an error.
            TimeoutError: If no response within timeout.
        """
        if not self._process or self._process.poll() is not None:
            raise RuntimeError("MCP server is not running")

        msg_id = self._next_id()
        request = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
        }
        if params is not None:
            request["params"] = params

        request_line = json.dumps(request) + "\n"

        with self._lock:
            self._process.stdin.write(request_line)
            self._process.stdin.flush()

            # Read response (blocking, line-delimited)
            response_line = self._process.stdout.readline()

        if not response_line:
            raise RuntimeError(f"MCP server closed connection (method: {method})")

        response = json.loads(response_line.strip())

        if "error" in response:
            err = response["error"]
            raise RuntimeError(
                f"MCP error ({err.get('code', '?')}): {err.get('message', 'Unknown')}"
            )

        return response.get("result", {})

    def _send_notification(self, method: str, params: dict | None = None):
        """Send a JSON-RPC 2.0 notification (no response expected)."""
        if not self._process or self._process.poll() is not None:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            notification["params"] = params

        notification_line = json.dumps(notification) + "\n"

        with self._lock:
            self._process.stdin.write(notification_line)
            self._process.stdin.flush()

    def _initialize(self):
        """Perform MCP protocol initialization handshake."""
        result = self._send_request("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {
                "name": "JadeAgent",
                "version": "0.2.0",
            },
        })

        logger.info(f"MCP server initialized: {result.get('serverInfo', {})}")

        # Send initialized notification
        self._send_notification("notifications/initialized")
        self._initialized = True

    def _discover_tools(self):
        """Discover available tools from the MCP server."""
        result = self._send_request("tools/list")
        self._raw_tools = result.get("tools", [])
        logger.info(f"Discovered {len(self._raw_tools)} MCP tools")

    def call_tool(self, name: str, arguments: dict | None = None) -> str:
        """
        Call a tool on the MCP server.

        Args:
            name: Tool name.
            arguments: Tool arguments dict.

        Returns:
            Tool result as a string.
        """
        result = self._send_request("tools/call", {
            "name": name,
            "arguments": arguments or {},
        })

        # Extract text from content array
        content = result.get("content", [])
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif item.get("type") == "image":
                    texts.append(f"[image: {item.get('mimeType', 'unknown')}]")
                else:
                    texts.append(json.dumps(item))
            elif isinstance(item, str):
                texts.append(item)

        return "\n".join(texts) if texts else json.dumps(result)

    @property
    def tool_names(self) -> list[str]:
        """List of discovered tool names."""
        return [t["name"] for t in self._raw_tools]

    @property
    def tool_schemas(self) -> list[dict]:
        """Raw MCP tool schemas."""
        return list(self._raw_tools)

    @property
    def tools(self) -> list:
        """
        Get MCP tools bridged to JadeAgent Tool format.

        Returns:
            List of Tool objects ready for Agent(tools=...).
        """
        from .bridge import bridge_mcp_tools
        return bridge_mcp_tools(self)

    def close(self):
        """Shut down the MCP server."""
        if self._process and self._process.poll() is None:
            try:
                self._process.stdin.close()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            logger.info("MCP server shut down")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    def __repr__(self) -> str:
        status = "running" if self._process and self._process.poll() is None else "stopped"
        return f"<MCPClient(tools={len(self._raw_tools)}, status={status})>"
