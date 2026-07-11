"""Request and response schemas for the pivotai inference API."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import SLM_FT_MODEL, SLM_DIST_MODEL, SLM_CURRICULUM_MODEL, SLM_BASELINE_MODEL

# Registry of valid model names — model param is validated against this set
# to prevent raw user input being passed to Ollama.
MODEL_REGISTRY: dict[str, str] = {
    SLM_FT_MODEL:         "Fine-tuned on Phase 1 synthetic pairs (SFT, Llama 3.1 8B)",
    SLM_DIST_MODEL:       "Distilled from Phase 2 DeepSeek agent traces (Llama 3.1 8B)",
    SLM_CURRICULUM_MODEL: "Curriculum-trained: Phase 1 → Phase 2 sequential (Llama 3.1 8B)",
    SLM_BASELINE_MODEL:   "Untuned base model — establishes pre-training floor (Llama 3.1 8B)",
}


class OptimizeRequest(BaseModel):
    persona: dict[str, Any]
    model: str = SLM_FT_MODEL

    @field_validator("model")
    @classmethod
    def model_must_be_registered(cls, v: str) -> str:
        if v not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{v}'. Valid options: {list(MODEL_REGISTRY)}"
            )
        return v


class OptimizeResponse(BaseModel):
    model: str
    output: dict[str, Any] | None
    raw_output: str
    latency_ms: int
    error: str | None = None


class ModelInfo(BaseModel):
    name: str
    description: str


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    models_available: list[str]
