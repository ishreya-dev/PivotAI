"""
Phase 2 output schemas.
TraceRecord is the canonical unit written to data/traces/agent_traces_*.jsonl.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]
    result: Any
    cache_hit: bool = False


class AgentStep(BaseModel):
    agent_name: str                     # "analyst" | "concierge" | "optimizer"
    tool_calls: list[ToolCall] = []
    reasoning: str                      # Agent's reasoning/response for this step
    output: dict[str, Any] = {}         # Structured output passed to next agent


class DayPlan(BaseModel):
    day: int
    location: str
    transit: str
    stay_district: str
    activities: str


class Itinerary(BaseModel):
    total_trip_cost: float
    daily_itinerary: list[DayPlan]


class GroundingStats(BaseModel):
    total_api_calls: int = 0
    cache_hits: int = 0
    savings_validated: bool = False


class TraceRecord(BaseModel):
    trace_id:          str   = Field(default_factory=lambda: str(uuid4()))
    generated_at:      str   = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    phase1_record_id:  str
    persona:           dict[str, Any]
    phase1_baseline:   dict[str, Any]       # original SyntheticRecord baseline itinerary
    agent_steps:       list[AgentStep] = []
    agent_optimized:   dict[str, Any] | None = None   # optimizer's final itinerary
    pivot_analysis:    str = ""
    savings_pct:       float = 0.0
    grounding:         GroundingStats = Field(default_factory=GroundingStats)