"""
Phase 4 — Eval suite data schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from uuid import uuid4
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvalMetrics(BaseModel):
    json_valid: bool
    savings_valid: bool | None = None
    budget_compliance: bool | None = None
    schema_compliance: float | None = None   # fraction of required keys present
    intent_alignment: float | None = None    # cosine sim via sentence-transformers
    rouge_l: float | None = None             # ROUGE-L vs Phase 1 teacher reference
    bertscore_f1: float | None = None        # BERTScore F1 vs Phase 1 teacher reference
    reasoning_coherence: float | None = None # LLM judge 0-1
    grounding_accuracy: float | None = None  # LLM judge 0-1
    judge_raw: str = ""


class EvalRecord(BaseModel):
    eval_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(default_factory=_now)
    model_name: str
    golden_record_id: str
    persona: dict
    prompt_used: str
    raw_output: str
    parsed_output: dict | None = None
    metrics: EvalMetrics


class RedTeamRecord(BaseModel):
    rt_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(default_factory=_now)
    prompt_type: str
    prompt: str
    model_name: str
    raw_output: str
    passed: bool
    judge_raw: str = ""
