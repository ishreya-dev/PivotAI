"""
Optimizer Agent — Step 3 (final) of the sequential chain.
Uses DeepSeek via OpenAI-compatible tool calling.

Synthesizes Analyst + Concierge outputs into a final optimized itinerary.
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

_SYSTEM = """You are the Optimizer Agent for ItinerAI-Bench, an AI travel optimizer.
Synthesize Analyst and Concierge data into a complete optimized day-by-day itinerary.

Requirements:
- Save at least the budget tier's minimum savings % from the baseline
- Keep the same number of days, destination, and traveler intents
- Use specific hotel names, transit modes, and activity names from prior agents

Return ONLY this JSON structure (nothing before or after the JSON):
{
  "optimized_itinerary": {
    "total_trip_cost": <INR number>,
    "daily_itinerary": [
      {"day": 1, "location": "...", "transit": "...", "stay_district": "...", "activities": "..."}
    ]
  },
  "pivot_analysis": "<100+ word explanation of every substitution and why it saves money>",
  "savings_pct": <float>,
  "pivots_made": [
    {"type": "transit|hotel|activity|dining", "baseline": "...", "optimized": "...", "saving_inr": int}
  ]
}"""


async def run_optimizer(persona: dict, baseline: dict, cost_report: dict,
                        substitutions: dict, adapter: MCPAdapter) -> AgentStep:
    budget = BUDGET_TIERS[persona["budget"]]
    min_savings = budget["min_savings_pct"]
    baseline_cost = baseline.get("total_trip_cost", 0)
    party_size = persona.get("size", {})
    total_people = party_size.get("adults", 1) + party_size.get("children", 0)
    activities_names = [a.get("name", "") for a in substitutions.get("activities", [])[:4]]

    user_msg = (
        f"Build optimized {persona['duration_days']}-day itinerary. "
        f"Trip: {persona['budget']} {persona['type']}, {persona['starting_city']} → "
        f"{persona['destination_city']}, {total_people} people. "
        f"Baseline: ₹{baseline_cost:,}. Must save ≥{min_savings}% "
        f"(target <₹{baseline_cost*(1-min_savings/100):,.0f}). "
        f"Intents: {', '.join(persona.get('intents', []))}.\n"
        f"Analyst: transit={cost_report.get('transit_recommendation','')}, "
        f"hotel={cost_report.get('hotel_recommendation','')}.\n"
        f"Concierge: activities={activities_names}, "
        f"stay={substitutions.get('stay_area_recommendation','city centre')}, "
        f"tips={substitutions.get('local_tips',[])[:2]}.\n"
        f"Baseline day 1: {json.dumps(baseline.get('daily_itinerary',[{}])[0])}.\n"
        f"Output ONLY the JSON now."
    )

    client, model = get_llm_client()
    # Optimizer uses all tools for any last-minute lookups
    tools = adapter.get_openai_tools()
    messages: list[dict] = [{"role": "user", "content": user_msg}]
    tool_calls: list[ToolCall] = []

    for _ in range(6):
        response = await llm_generate(client, model, messages, tools, _SYSTEM)
        msg = response.choices[0].message
        messages.append(msg_to_dict(msg))

        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = await adapter.call_tool(tc.function.name, args)
            tool_calls.append(ToolCall(name=tc.function.name, args=args, result=result))
            log.info("Optimizer tool call", tool=tc.function.name)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                              "content": json.dumps(result)})

    reasoning = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
    result = extract_json(reasoning)

    # If Llama didn't produce the itinerary, retry with a tighter, JSON-only prompt
    if not result.get("optimized_itinerary"):
        log.warning("Optimizer incomplete — retrying with JSON-only prompt")
        retry_msg = (
            f"Output ONLY valid JSON (no text before or after) for a "
            f"{persona['duration_days']}-day optimized itinerary: "
            f"{persona['starting_city']} → {persona['destination_city']}, "
            f"{persona['budget']}, {total_people} people, baseline ₹{baseline_cost:,}, "
            f"save ≥{min_savings}%, use {cost_report.get('transit_recommendation','train')}, "
            f"activities: {', '.join(activities_names) or 'local sights'}."
        )
        retry_resp = await llm_generate(
            client, model,
            [{"role": "user", "content": retry_msg}],
            None,   # no tools — pure generation
            _SYSTEM,
        )
        reasoning = retry_resp.choices[0].message.content or ""
        result = extract_json(reasoning)
        log.info("Optimizer retry", json_keys=list(result.keys()))

    opt_cost = result.get("optimized_itinerary", {}).get("total_trip_cost", baseline_cost)
    actual_savings = ((baseline_cost - opt_cost) / baseline_cost * 100) if baseline_cost else 0.0
    if "savings_pct" not in result:
        result["savings_pct"] = round(actual_savings, 2)

    log.info("Optimizer complete", savings_pct=result.get("savings_pct"),
             tool_calls=len(tool_calls))
    return AgentStep(agent_name="optimizer", tool_calls=tool_calls,
                     reasoning=reasoning, output=result)