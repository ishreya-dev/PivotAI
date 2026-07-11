"""
Concierge Agent — Step 2 of the sequential chain.
Uses DeepSeek via OpenAI-compatible tool calling.

Finds real POIs, restaurants, and web tips matching the traveler's intents.
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

_CONCIERGE_TOOLS = ["search_pois", "search_restaurants", "web_search"]

_SYSTEM = """You are the Concierge Agent for ItinerAI-Bench, an AI travel optimizer.
Find experience-equivalent but cheaper activities, restaurants, and local tips.

You MUST call:
1. search_pois(city, intent, price_level) — for EACH traveler intent
2. search_restaurants(city, price_level) — for the destination
3. web_search — at least once for budget tips

Return substitutions JSON (end your response with it):
{
  "destination_city": str,
  "activities": [{"name": str, "type": str, "cost_free_or_paid": str, "saves_vs_baseline": str}],
  "restaurants": [{"name": str, "cuisine": str, "price_level": int, "address": str}],
  "local_tips": [str],
  "stay_area_recommendation": str,
  "total_activity_savings_estimate_inr": int
}"""


async def run_concierge(persona: dict, cost_report: dict, adapter: MCPAdapter) -> AgentStep:
    budget = BUDGET_TIERS[persona["budget"]]
    dest = persona["destination_city"]
    intents = persona.get("intents", ["Cultural"])
    price_level = budget["price_level"]

    user_msg = (
        f"Find budget-optimized activities and dining for {persona['budget']} trip to {dest}. "
        f"Intents: {', '.join(intents)}. price_level={price_level}. "
        f"Cost drivers from Analyst: {cost_report.get('cost_drivers', [])}. "
        f"Call search_pois for each intent. Call search_restaurants('{dest}', {price_level}). "
        f"Call web_search('budget {intents[0].lower()} tips {dest} 2024'). "
        f"Return substitutions JSON."
    )

    client, model = get_llm_client()
    tools = adapter.get_openai_tools(_CONCIERGE_TOOLS)
    messages: list[dict] = [{"role": "user", "content": user_msg}]
    tool_calls: list[ToolCall] = []

    for _ in range(10):
        response = await llm_generate(client, model, messages, tools, _SYSTEM)
        msg = response.choices[0].message
        messages.append(msg_to_dict(msg))

        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = await adapter.call_tool(tc.function.name, args)
            tool_calls.append(ToolCall(name=tc.function.name, args=args, result=result))
            log.info("Concierge tool call", tool=tc.function.name, args=str(args)[:100])
            messages.append({"role": "tool", "tool_call_id": tc.id,
                              "content": json.dumps(result)})

    reasoning = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
    substitutions = extract_json(reasoning)

    log.info("Concierge complete", tool_calls=len(tool_calls),
             activities=len(substitutions.get("activities", [])))
    return AgentStep(agent_name="concierge", tool_calls=tool_calls,
                     reasoning=reasoning, output=substitutions)