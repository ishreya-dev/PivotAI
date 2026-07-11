# Phase 2 — Multi-Agent Orchestration

Runs a 3-agent pipeline (Analyst → Concierge → Optimizer) over Phase 1 personas using real MCP tool servers. Produces grounded reasoning traces used as the distillation training signal for Phase 3.

## Final Output

**Canonical file**: `data/traces/agent_traces_all.jsonl`

| Metric | Value |
|--------|-------|
| Quality traces | 500 |
| Avg savings | 20.0% |
| Savings range | 3% – 94% |
| Unique personas covered | 500 |
| Agent LLM | DeepSeek V4 Flash (`deepseek-chat`) |
| Avg API calls per trace | ~2,900 |

---

## Architecture

```
phase2_agents/run.py  (CLI entrypoint)
        │
        ▼
  Supervisor  ──  opens async connections to all 4 MCP servers
        │
  MCPAdapter  ──  exposes MCP tools in OpenAI function-calling format
        │
  [Analyst]   ──  get_route, search_hotels, search_flights → cost_report
        │
  [Concierge] ──  search_pois, search_restaurants, web_search → substitutions
        │
  [Optimizer] ──  all tools (final lookups) → optimized itinerary + pivot_analysis
        │
        ▼
  TraceRecord  →  data/traces/agent_traces_all.jsonl
```

---

## MCP Servers

All 4 servers implement the official `mcp` Python library (SSE transport on localhost). They plug directly into Claude Desktop/Code without modification.

| Server | Port | Data Source | Tools |
|--------|------|-------------|-------|
| `routing_server.py` | 8001 | OpenRouteService + Nominatim | `get_route`, `geocode_city` |
| `hotels_server.py` | 8002 | Overpass API (OSM) + haversine | `search_hotels`, `search_flights` |
| `overpass_server.py` | 8003 | Overpass API (OSM) | `search_pois`, `search_restaurants` |
| `search_server.py` | 8004 | DuckDuckGo | `web_search` |

> `hotels_server.py` uses Overpass (OSM) for real hotel data and a haversine formula for flight cost estimates. No external API key required.

---

## Agent Chain

| Agent | Role | Tools used | Output |
|-------|------|-----------|--------|
| Analyst | Identifies transit + hotel cost drivers vs baseline | `get_route`, `search_hotels`, `search_flights` | `cost_report` dict |
| Concierge | Finds cheaper POIs + dining matching traveler intents | `search_pois`, `search_restaurants`, `web_search` | `substitutions` dict |
| Optimizer | Synthesizes final day-by-day itinerary + pivot analysis | all tools | `agent_optimized` + `pivot_analysis` |

---

## Trace Schema

```json
{
  "trace_id": "uuid4",
  "generated_at": "ISO-8601",
  "phase1_record_id": "uuid4",
  "persona": { "starting_city", "destination_city", "type", "size", "intents", "budget", "duration_days" },
  "phase1_baseline": { "total_trip_cost": 0, "daily_itinerary": [] },
  "agent_steps": [
    {
      "agent_name": "analyst",
      "tool_calls": [ { "name", "args", "result", "cache_hit" } ],
      "reasoning": "DeepSeek's full reasoning text for this step",
      "output": { "cost_report": {} }
    },
    { "agent_name": "concierge", "..." },
    { "agent_name": "optimizer", "..." }
  ],
  "agent_optimized": { "total_trip_cost": 0, "daily_itinerary": [] },
  "pivot_analysis": "explanation of substitutions and savings",
  "savings_pct": 15.5,
  "grounding": { "total_api_calls": 2900, "cache_hits": 1800, "savings_validated": true }
}
```

---

## Running the Agent Pipeline

Requires `DEEPSEEK_API_KEY` in `.env`.

```bash
# Terminal 1–4: start MCP servers
python phase2_agents/mcp_servers/routing_server.py
python phase2_agents/mcp_servers/hotels_server.py
python phase2_agents/mcp_servers/overpass_server.py
python phase2_agents/mcp_servers/search_server.py

# Terminal 5: run agents
python phase2_agents/run.py --limit 25 --concurrency 3 --verbose

# Auto-resume after interruption — skips already-processed record IDs automatically
python phase2_agents/run.py --limit 500 --concurrency 3

# Debug one specific Phase 1 record
python phase2_agents/run.py --record-id <uuid> --verbose
```

---

## Data Quality

The canonical trace file contains only quality records:
- `agent_optimized` is not None (optimizer completed successfully)
- `len(agent_steps) >= 2` (at least 2 agents ran)
- One trace per unique persona (best run kept when same persona was processed multiple times)
- `api_calls >= 50` (rules out early incomplete test runs with 7–15 calls)

Filtering is applied by `phase3_training/prepare_distill.py` before training.

---

## Key Design Decisions

**Why DeepSeek V4 Flash?** OpenAI-compatible API, strong function-calling support, and cost $4 for 500 multi-agent traces — exactly matching the Phase 1 GPT-4o-mini budget so the fine-tune vs. distill comparison is not confounded by data cost. DeepSeek uses the standard OpenAI tool-calling format, making the agent code provider-agnostic and the same key is reused for Phase 4 eval judging.

**Why MCP over direct API calls?** MCP servers are reusable across agents, sessions, and tools (Claude Desktop, Claude Code, custom agents). The same 4 servers required for Phase 2 can be reused in Phase 5 UI for live itinerary grounding.

**Why cache MCP responses?** Routing queries between the same 20 cities repeat constantly. `@api_cache(ttl=86400)` in each server collapses thousands of runs to ~40 unique Overpass/ORS calls per day — well within all free tier limits.

**Why Overpass for hotels?** Overpass API (OpenStreetMap) provides real hotel names, star ratings, and addresses — no API key needed, no rate limits, and comprehensive coverage of Indian cities. Several flight pricing APIs were evaluated but all required paid plans or had rate limits too low for 500 concurrent agent runs.

---

## Log Files

| File | Contents |
|------|---------|
| `logs/phase2/agents.log` | Per-agent tool calls and reasoning (structured JSON) |
| `logs/phase2/mcp_servers.log` | MCP server requests, cache hits, Overpass/ORS errors |
| `logs/phase2/progress.log` | Per-record status: savings%, API calls, duration |