"""
Single place to create the LLM client and pick the model.
All three agents import from here — change PHASE2_PROVIDER in config.py to switch.

Providers:
  "deepseek" → DeepSeek V4 Flash (active) — paid, $4 for 500 multi-agent traces
  "groq"     → Groq (Llama 3.3 70B) — free tier, ~12h for 500 records
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEEPSEEK_AGENT_MODEL, DEEPSEEK_BASE_URL,
    PHASE2_PROVIDER,
)
from phase2_agents.llm_utils import _rate_limiter_fast, _rate_limiter_groq
import phase2_agents.llm_utils as _gu


def get_llm_client():
    """Return (openai.OpenAI client, model_name) for the configured provider."""
    from openai import OpenAI

    provider = PHASE2_PROVIDER.lower()

    if provider == "deepseek":
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY not set in .env")
        _gu._rate_limiter = _rate_limiter_fast
        return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL), DEEPSEEK_AGENT_MODEL

    elif provider == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        _gu._rate_limiter = _rate_limiter_groq
        return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1"), "llama-3.3-70b-versatile"

    else:
        raise ValueError(f"Unknown PHASE2_PROVIDER: '{provider}'. Use: deepseek | groq")