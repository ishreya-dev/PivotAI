"""
pivotai Inference API — FastAPI server wrapping Ollama.

Endpoints:
  GET  /health          — Ollama ping + registered model availability
  GET  /models          — list registered pivotai models with descriptions
  POST /optimize        — run inference on a persona dict
  GET  /results/summary — latest eval summary JSON (no inference, instant)
  GET  /results/compare — head-to-head win rates from eval results

Start:
  uvicorn phase5_serving.api.main:app --reload --port 8000

Docs:
  http://localhost:8000/docs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import EVALS_DIR
from phase5_serving.api.schemas import (
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    MODEL_REGISTRY,
    OptimizeRequest,
    OptimizeResponse,
)
from phase5_serving.api.ollama_client import generate, is_reachable, list_running_models

app = FastAPI(
    title="pivotai Inference API",
    description=(
        "REST interface for pivotai SLMs — three fine-tuned Llama 3.1 8B variants "
        "trained to find Price-Pivot Points in Indian domestic travel itineraries."
    ),
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse, summary="Ollama health check")
async def health() -> HealthResponse:
    """Check whether Ollama is running and which registered models are available."""
    reachable = await is_reachable()
    available: list[str] = []
    if reachable:
        running = await list_running_models()
        running_set = {m.split(":")[0] for m in running}
        available = [m for m in MODEL_REGISTRY if m.split(":")[0] in running_set or m in running]
    return HealthResponse(
        status="ok" if reachable else "ollama_unreachable",
        ollama_reachable=reachable,
        models_available=available,
    )


@app.get("/models", response_model=ModelsResponse, summary="List registered models")
async def models() -> ModelsResponse:
    """Return the four registered pivotai models and their training descriptions."""
    return ModelsResponse(
        models=[ModelInfo(name=k, description=v) for k, v in MODEL_REGISTRY.items()]
    )


@app.post("/optimize", response_model=OptimizeResponse, summary="Run inference on a persona")
async def optimize(req: OptimizeRequest) -> OptimizeResponse:
    """
    Generate an optimized travel itinerary for the given persona.

    The persona dict must match the pivotai format:
    ```json
    {
      "starting_city": "Mumbai",
      "destination_city": "Delhi",
      "type": "Solo",
      "size": {"adults": 1, "children": 0},
      "intents": ["Adventure"],
      "budget": "Shoestring",
      "duration_days": 5,
      "duration_nights": 4
    }
    ```

    Inference runs locally via Ollama — expect 30–120 s on MacBook Air CPU.
    """
    try:
        raw, latency_ms = await generate(req.model, req.persona)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ollama error: {exc}") from exc

    parsed: dict | None = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        pass

    return OptimizeResponse(
        model=req.model,
        output=parsed,
        raw_output=raw,
        latency_ms=latency_ms,
    )


@app.get("/results/summary", summary="Latest eval summary (no inference)")
async def results_summary() -> JSONResponse:
    """Return the most recent eval summary JSON from data/evals/."""
    candidates = sorted(EVALS_DIR.glob("summary_*.json"))
    if not candidates:
        raise HTTPException(status_code=404, detail="No eval summary found. Run phase4_evals/compare.py first.")
    data = json.loads(candidates[-1].read_text(encoding="utf-8"))
    return JSONResponse(content=data)


@app.get("/results/compare", summary="Head-to-head win rates")
async def results_compare() -> JSONResponse:
    """Return head-to-head win rates from the latest eval summary."""
    candidates = sorted(EVALS_DIR.glob("summary_*.json"))
    if not candidates:
        raise HTTPException(status_code=404, detail="No eval summary found.")
    data = json.loads(candidates[-1].read_text(encoding="utf-8"))
    return JSONResponse(content={
        "head_to_head": data.get("head_to_head", {}),
        "generated_at": data.get("generated_at"),
    })
