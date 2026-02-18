#!/usr/bin/env python3
"""
Simple MCP server for testing JadeAgent's MCP client.

Implements a minimal MCP server with stdio transport and 3 test tools:
- echo: Returns the input text
- calculator: Evaluates math expressions
- get_time: Returns current time

Usage:
    python examples/test_mcp_server.py

This server is used for testing. Launch it via MCPClient:
    mcp = MCPClient("python examples/test_mcp_server.py")
"""

import json
import sys
from datetime import datetime


def handle_request(request: dict) -> dict | None:
    """Handle a JSON-RPC 2.0 request and return response."""
    method = request.get("method", "")
    msg_id = request.get("id")
    params = request.get("params", {})

    # Notifications (no id) → no response
    if msg_id is None:
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "JadeTestServer",
                    "version": "1.0.0",
                },
            },
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo back the input text",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "Text to echo back",
                                },
                            },
                            "required": ["text"],
                        },
                    },
                    {
                        "name": "calculator",
                        "description": "Calculate a math expression safely",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Math expression to evaluate",
                                },
                            },
                            "required": ["expression"],
                        },
                    },
                    {
                        "name": "get_time",
                        "description": "Get the current date and time",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                ],
            },
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name == "echo":
            text = arguments.get("text", "")
            content = [{"type": "text", "text": f"Echo: {text}"}]

        elif tool_name == "calculator":
            expr = arguments.get("expression", "")
            try:
                result = eval(expr, {"__builtins__": {}}, {})
                content = [{"type": "text", "text": str(result)}]
            except Exception as e:
                content = [{"type": "text", "text": f"Error: {e}"}]

        elif tool_name == "get_time":
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            content = [{"type": "text", "text": now}]

        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"content": content},
        }

    else:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def main():
    """Run the MCP server, reading JSON-RPC from stdin, writing to stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        response = handle_request(request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
