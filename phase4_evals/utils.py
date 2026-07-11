"""
Phase 4 — shared helpers used by generate_responses, score_responses, and red_team.
Single source of truth for prompt format, Ollama call, and num_predict settings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SLM_FT_MODEL, SLM_DIST_MODEL, SLM_CURRICULUM_MODEL, SLM_BASELINE_MODEL, OLLAMA_BASE_URL

# ── Prompt instructions ───────────────────────────────────────────────────────

FT_INSTRUCTION = (
    "Act as pivotai Optimizer. Given a traveler persona for an Indian domestic trip, "
    "produce an optimized day-by-day itinerary that minimizes total cost while respecting "
    "the budget tier, trip type, and traveler intents. Identify the primary Price-Pivot Point "
    "(transit, accommodation, or activity substitution that saves ≥5%) and explain it clearly."
)

DISTILL_INSTRUCTION = (
    "Act as pivotai Supervisor for an Indian domestic trip. Coordinate the Analyst, "
    "Concierge, and Optimizer agents to find Price-Pivot Points and produce an optimized "
    "itinerary. Show the reasoning chain for each agent handoff, then provide the final "
    "pivot analysis and optimized itinerary."
)

# ── Token budget per model ────────────────────────────────────────────────────
# distill outputs a reasoning chain before JSON — needs more tokens.
# ft/curriculum output compact JSON — 512 stays well under 600s timeout on CPU.

NUM_PREDICT: dict[str, int] = {
    SLM_DIST_MODEL: 1024,
    SLM_FT_MODEL: 512,
    SLM_CURRICULUM_MODEL: 512,
    SLM_BASELINE_MODEL: 512,
}

# ── Shared functions ──────────────────────────────────────────────────────────

def build_prompt(persona: dict, model_name: str) -> str:
    """Build Alpaca-format prompt. Must match training format exactly."""
    instruction = DISTILL_INSTRUCTION if model_name == SLM_DIST_MODEL else FT_INSTRUCTION
    persona_str = json.dumps(persona, ensure_ascii=False)
    return f"### Instruction:\n{instruction}\n\n### Input:\n{persona_str}\n\n### Response:\n"


def call_ollama(model_name: str, prompt: str, timeout: int = 600) -> str:
    """POST to local Ollama /api/generate. Raises on timeout or HTTP error."""
    resp = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": NUM_PREDICT.get(model_name, 512),
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")
