"""Async Ollama client for the inference API."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import OLLAMA_BASE_URL
from phase4_evals.utils import build_prompt, NUM_PREDICT

INFERENCE_TIMEOUT = int(os.getenv("INFERENCE_TIMEOUT", "600"))


async def generate(model_name: str, persona: dict) -> tuple[str, int]:
    """
    Send a persona to Ollama and return (raw_output, latency_ms).
    Uses the same Alpaca prompt format as the eval pipeline for consistency.
    """
    prompt = build_prompt(persona, model_name)
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": NUM_PREDICT.get(model_name, 512),
        },
    }

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        resp.raise_for_status()

    latency_ms = int((time.monotonic() - t0) * 1000)
    raw = resp.json().get("response", "")
    return raw, latency_ms


async def list_running_models() -> list[str]:
    """Return names of models currently loaded in Ollama."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


async def is_reachable() -> bool:
    """Ping Ollama health endpoint."""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False
