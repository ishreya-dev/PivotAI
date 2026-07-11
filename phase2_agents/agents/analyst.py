"""
Analyst Agent — Step 1 of the sequential chain.
Uses DeepSeek via OpenAI-compatible tool calling.

Given a persona + Phase 1 baseline itinerary, the Analyst:
  1. Queries routing tools to find actual travel distances and transit costs
  2. Queries hotels for real prices at the destination
  3. Identifies the biggest cost drivers in the baseline
  4. Returns a cost_report dict for the Concierge to use
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BUDGET_TIERS
from phase2_agents.llm_client import get_llm_client
from phase2_agents.llm_utils import llm_generate, msg_to_dict
from phase2_agents.json_utils import extract_json
from phase2_agents.mcp_adapter import MCPAdapter
from phase2_agents.schemas import AgentStep, ToolCall
from utils.logger import get_logger

log = get_logger("phase2", "agents")

_ANALYST_TOOLS = ["get_route", "geocode_city", "search_hotels", "search_flights"]

_SYSTEM = """You are the Analyst Agent for ItinerAI-Bench, an AI travel optimizer.
Identify Price-Pivot Points — places where real pricing data shows the baseline overspends.

You MUST call:
1. get_route(origin_city, dest_city, mode) — twice: baseline mode AND cheapest alternative
2. search_hotels(city, stars, nights) — for the destination with the correct star rating

Return a cost_report JSON (end your response with it):
{
  "transit_baseline_cost_per_person": int,
  "transit_optimized_cost_per_person": int,
  "transit_savings_pct": float,
  "transit_recommendation": str,
  "hotel_baseline_cost_per_night": int,
  "hotel_options": [...],
  "hotel_savings_pct": float,
  "hotel_recommendation": str,
  "total_estimated_savings_pct": float,
  "cost_drivers": [str]
}"""


async def run_analyst(persona: dict, baseline: dict, adapter: MCPAdapter) -> AgentStep:
    budget = BUDGET_TIERS[persona["budget"]]
    origin, dest = persona["starting_city"], persona["destination_city"]
    nights = persona.get("duration_nights", persona.get("duration_days", 1) - 1)
    baseline_cost = baseline.get("total_trip_cost", 0)
    party_size = persona.get("size", {})
    total_people = party_size.get("adults", 1) + party_size.get("children", 0)

    day1 = baseline.get("daily_itinerary", [{}])[0]
    t = day1.get("transit", "train").lower()
    baseline_mode = "flight" if "flight" in t or "fly" in t else "train" if "train" in t else "bus"
    alt_mode = "train" if baseline_mode == "flight" else ("flight" if baseline_cost > 50000 else "bus")

    user_msg = (
        f"Analyze this {persona['budget']} {persona['type']} trip: {origin} → {dest}, "
        f"{persona['duration_days']} days, {total_people} people. "
        f"Budget tier: {persona['budget']} ({budget['hotel_stars']}★ hotel, "
        f"₹{budget['min_daily']:,}–₹{budget['max_daily']:,}/person/day). "
        f"Baseline transit: {baseline_mode}, baseline cost: ₹{baseline_cost:,}. "
        f"Call get_route for {baseline_mode} AND {alt_mode}. "
        f"Call search_hotels({budget['hotel_stars']} stars, {nights} nights). "
        f"Return cost_report JSON."
    )

    client, model = get_llm_client()
    tools = adapter.get_openai_tools(_ANALYST_TOOLS)
    messages: list[dict] = [{"role": "user", "content": user_msg}]
    tool_calls: list[ToolCall] = []

    for _ in range(8):
        response = await llm_generate(client, model, messages, tools, _SYSTEM)
        msg = response.choices[0].message
        messages.append(msg_to_dict(msg))

        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = await adapter.call_tool(tc.function.name, args)
            tool_calls.append(ToolCall(name=tc.function.name, args=args, result=result))
            log.info("Analyst tool call", tool=tc.function.name, args=str(args)[:100])
            messages.append({"role": "tool", "tool_call_id": tc.id,
                              "content": json.dumps(result)})

    reasoning = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
    cost_report = extract_json(reasoning)

    log.info("Analyst complete", tool_calls=len(tool_calls),
             savings_est=cost_report.get("total_estimated_savings_pct", 0))
    return AgentStep(agent_name="analyst", tool_calls=tool_calls,
                     reasoning=reasoning, output=cost_report)