"""
Supervisor — orchestrates the sequential Analyst → Concierge → Optimizer chain.

Accepts an optional shared MCPAdapter so run.py can open ONE set of SSE connections
for all concurrent workers, preventing file-descriptor exhaustion.
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BUDGET_TIERS, MCP_SERVERS
from phase2_agents.agents.analyst import run_analyst
from phase2_agents.agents.concierge import run_concierge
from phase2_agents.agents.optimizer import run_optimizer
from phase2_agents.mcp_adapter import MCPAdapter
from phase2_agents.schemas import GroundingStats, TraceRecord
from utils.logger import get_logger

log = get_logger("phase2", "agents")


async def run_trace(
    phase1_record: dict[str, Any],
    adapter: MCPAdapter | None = None,
) -> TraceRecord:
    """
    Run the full Analyst → Concierge → Optimizer chain for one Phase 1 record.

    adapter: pass a shared MCPAdapter from the caller to reuse SSE connections.
             If None, a fresh adapter is created (and closed) for this record.
    """
    record_id = phase1_record.get("id", "unknown")
    persona   = phase1_record["persona"]
    baseline  = phase1_record["pair"]["baseline"]

    log.info("Trace start", record_id=record_id,
             origin=persona["starting_city"], dest=persona["destination_city"])

    async def _run(adp: MCPAdapter) -> TraceRecord:
        log.info("Running Analyst",   record_id=record_id)
        analyst_step   = await run_analyst(persona, baseline, adp)

        log.info("Running Concierge", record_id=record_id)
        concierge_step = await run_concierge(persona, analyst_step.output, adp)

        log.info("Running Optimizer", record_id=record_id)
        optimizer_step = await run_optimizer(
            persona, baseline,
            analyst_step.output,
            concierge_step.output,
            adp,
        )

        stats = adp.stats
        optimizer_output = optimizer_step.output
        savings_pct      = optimizer_output.get("savings_pct", 0.0)

        record = TraceRecord(
            phase1_record_id=record_id,
            persona=persona,
            phase1_baseline=baseline,
            agent_steps=[analyst_step, concierge_step, optimizer_step],
            agent_optimized=optimizer_output.get("optimized_itinerary"),
            pivot_analysis=optimizer_output.get("pivot_analysis", ""),
            savings_pct=savings_pct,
            grounding=GroundingStats(
                total_api_calls=stats["total_api_calls"],
                cache_hits=stats["cache_hits"],
                savings_validated=savings_pct >= BUDGET_TIERS[persona["budget"]]["min_savings_pct"],
            ),
        )

        log.info("Trace complete", record_id=record_id, savings_pct=savings_pct,
                 api_calls=stats["total_api_calls"], validated=record.grounding.savings_validated)
        return record

    if adapter is not None:
        return await _run(adapter)

    # No shared adapter provided — create one just for this record
    async with MCPAdapter(MCP_SERVERS) as fresh_adapter:
        return await _run(fresh_adapter)