"""
pivotai MCP Search Server — port 8004
Wraps DuckDuckGo web search for travel context queries.
No API key required.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MCP_SERVERS
from utils.cache import api_cache
from utils.logger import get_logger

load_dotenv()
log = get_logger("phase2", "mcp_servers")
_PORT = int(MCP_SERVERS["search"].split(":")[-1])
mcp = FastMCP("pivotai-search", host="0.0.0.0", port=_PORT)


@api_cache(ttl=86400)
def _search_cached(query: str, max_results: int) -> list[dict]:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
            for r in results
        ]
    except Exception as exc:
        log.warning("Web search failed", query=query, error=str(exc))
        return []


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> dict:
    """
    Search the web for travel-related information using DuckDuckGo.
    Useful for current hotel prices, local tips, seasonal info, and travel advisories.
    Returns a list of results with title, snippet, and URL.
    max_results: 1–10 (default 5)
    """
    max_results = max(1, min(max_results, 10))
    results = _search_cached(query, max_results)
    log.info("web_search", query=query, results_returned=len(results))
    return {"query": query, "results": results}


if __name__ == "__main__":
    log.info("Starting pivotai search MCP server", port=_PORT)
    mcp.run(transport="sse")
