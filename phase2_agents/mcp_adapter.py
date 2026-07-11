"""
MCPAdapter — bridges MCP servers to LLM tool calling.

Usage:
    async with MCPAdapter(MCP_SERVERS) as adapter:
        tools  = adapter.get_openai_tools(["get_route", "geocode_city"])
        result = await adapter.call_tool("get_route", {"origin_city": "Delhi", ...})

Design:
    - Connects to all 4 MCP servers via SSE on startup
    - Builds a tool registry: name → (server_name, MCP Tool definition)
    - Exposes tools in OpenAI-compatible format for any OpenAI-compatible LLM
    - Routes call_tool() to the correct server session
    - Tracks cache hits via a 'cache_hit' heuristic (< 50ms = cache hit)
"""

import json
import time
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MCP_SERVERS

from utils.logger import get_logger

log = get_logger("phase2", "agents")


class MCPAdapter:
    """
    Async context manager that connects to MCP servers and exposes
    tools in OpenAI-compatible format for any OpenAI-compatible LLM.
    """

    def __init__(self, server_urls: dict[str, str] | None = None):
        self._urls = server_urls or MCP_SERVERS
        self._stack = AsyncExitStack()
        # Maps tool_name → (server_name, mcp_Tool_object)
        self._registry: dict[str, tuple[str, Any]] = {}
        # Maps server_name → ClientSession
        self._sessions: dict[str, Any] = {}
        self._api_call_count = 0
        self._cache_hit_count = 0

    async def __aenter__(self) -> "MCPAdapter":
        await self._stack.__aenter__()
        for name, url in self._urls.items():
            sse_url = url.rstrip("/") + "/sse"
            try:
                from mcp import ClientSession
                from mcp.client.sse import sse_client

                streams = await self._stack.enter_async_context(sse_client(sse_url))
                session = await self._stack.enter_async_context(ClientSession(*streams))
                await session.initialize()
                self._sessions[name] = session

                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    self._registry[tool.name] = (name, tool)
                    log.info("MCP tool registered", server=name, tool=tool.name)
            except Exception as exc:
                log.warning("Cannot connect to MCP server", server=name, url=sse_url, error=str(exc))
        return self

    async def __aexit__(self, *args) -> None:
        await self._stack.__aexit__(*args)

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Call a named MCP tool and return the parsed result dict."""
        entry = self._registry.get(name)
        if not entry:
            return {"error": f"Unknown tool: {name}"}

        server_name, _ = entry
        session = self._sessions.get(server_name)
        if not session:
            return {"error": f"Server {server_name} not connected"}

        self._api_call_count += 1
        t0 = time.monotonic()
        try:
            result = await session.call_tool(name, args)
            elapsed_ms = (time.monotonic() - t0) * 1000
            # Heuristic: very fast responses are cache hits in the server
            is_cache_hit = elapsed_ms < 50
            if is_cache_hit:
                self._cache_hit_count += 1

            if result.content and hasattr(result.content[0], "text"):
                try:
                    return json.loads(result.content[0].text)
                except json.JSONDecodeError:
                    return {"result": result.content[0].text}
            return {}
        except Exception as exc:
            log.warning("MCP tool call failed", tool=name, error=str(exc))
            return {"error": str(exc)}

    def get_openai_tools(self, tool_names: list[str] | None = None) -> list[dict]:
        """Return tools in OpenAI/Groq format (list of {type, function} dicts)."""
        tools = []
        for name, (_, tool_def) in self._registry.items():
            if tool_names is None or name in tool_names:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool_def.name,
                        "description": tool_def.description or tool_def.name,
                        "parameters": tool_def.inputSchema or {"type": "object", "properties": {}},
                    },
                })
        return tools

    @property
    def stats(self) -> dict:
        return {
            "total_api_calls": self._api_call_count,
            "cache_hits":      self._cache_hit_count,
            "tools_available": list(self._registry.keys()),
        }