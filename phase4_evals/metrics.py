"""
Phase 4 — Eval metrics. Pure functions, no side effects.
All external API calls are rate-limited and disk-cached.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BUDGET_TIERS, VALIDATION_COST_MARGIN, DEEPSEEK_AGENT_MODEL, DEEPSEEK_BASE_URL
from utils.cache import api_cache
from utils.logger import get_logger

log = get_logger("phase4", "metrics")

_sentence_model = None


def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_model


# ── JSON parsing ─────────────────────────────────────────────────────────────

def check_json_valid(raw: str) -> tuple[bool, dict | None]:
    """Try direct parse, then extract last JSON block from prose (for distill outputs)."""
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return True, parsed
    except json.JSONDecodeError:
        pass

    # Regex fallback: find the last {...} block in the text
    matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL))
    for m in reversed(matches):
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, dict):
                return True, parsed
        except json.JSONDecodeError:
            continue

    return False, None


# ── Local metrics (no API) ────────────────────────────────────────────────────

def check_savings_valid(parsed: dict, persona: dict) -> bool | None:
    """Did the model find a pivot that saves >= the tier minimum?"""
    budget = persona.get("budget", "")
    tier = BUDGET_TIERS.get(budget)
    if not tier:
        return None
    min_pct = tier["min_savings_pct"]

    # Try direct savings_pct field
    savings_pct = parsed.get("savings_pct") or parsed.get("transit_savings_pct")
    if savings_pct is not None:
        try:
            return float(savings_pct) >= min_pct
        except (ValueError, TypeError):
            pass

    # Try to compute from baseline vs optimized cost
    try:
        baseline = float(
            parsed.get("baseline_cost")
            or parsed.get("total_baseline_cost")
            or _dig(parsed, "baseline", "total_trip_cost")
            or 0
        )
        optimized = float(
            parsed.get("optimized_cost")
            or _dig(parsed, "optimized", "total_trip_cost")
            or 0
        )
        if baseline > 0 and optimized > 0 and optimized < baseline:
            pct = (baseline - optimized) / baseline * 100
            return pct >= min_pct
    except (ValueError, TypeError):
        pass

    return None


def check_budget_compliance(parsed: dict, persona: dict) -> bool | None:
    """Is the optimized cost within the persona's budget tier bounds (±20%)?"""
    budget = persona.get("budget", "")
    tier = BUDGET_TIERS.get(budget)
    if not tier:
        return None

    duration = persona.get("duration_days", 0)
    size = persona.get("size", {})
    party = (size.get("adults", 1) or 1) + (size.get("children", 0) or 0)
    if duration <= 0 or party <= 0:
        return None

    expected_min = tier["min_daily"] * duration * party * (1 - VALIDATION_COST_MARGIN)
    expected_max = tier["max_daily"] * duration * party * (1 + VALIDATION_COST_MARGIN)

    cost = (
        parsed.get("optimized_cost")
        or _dig(parsed, "optimized", "total_trip_cost")
    )
    if cost is None:
        return None
    try:
        cost = float(cost)
    except (ValueError, TypeError):
        return None

    return expected_min <= cost <= expected_max


def score_intent_alignment(persona: dict, parsed: dict) -> float | None:
    """Cosine similarity between persona intents and itinerary activities (sentence-transformers)."""
    intents = persona.get("intents", [])
    if not intents:
        return None

    # Collect activity text from itinerary
    activities_text = _extract_activities(parsed)
    if not activities_text:
        return None

    try:
        model = _get_sentence_model()
        from sentence_transformers.util import cos_sim
        intent_text = ", ".join(intents)
        vecs = model.encode([intent_text, activities_text], convert_to_tensor=True)
        score = float(cos_sim(vecs[0], vecs[1]).item())
        return round(max(0.0, min(1.0, score)), 4)
    except Exception as exc:
        log.warning("Intent alignment failed", error=str(exc))
        return None


# ── Reference-based metrics (ROUGE-L, BERTScore, schema compliance) ──────────

# Required keys for a valid pivotai output
_TOP_LEVEL_KEYS = {"optimized", "pivot_analysis"}
_OPTIMIZED_KEYS = {"total_trip_cost", "daily_itinerary"}

_bertscore_model_type = "distilbert-base-uncased"  # 260MB, fast CPU inference


def check_schema_compliance(parsed: dict) -> float | None:
    """Fraction of required keys present: 0.0–1.0."""
    if not parsed:
        return 0.0
    total = len(_TOP_LEVEL_KEYS) + len(_OPTIMIZED_KEYS)
    found = sum(1 for k in _TOP_LEVEL_KEYS if k in parsed)
    optimized = parsed.get("optimized", {})
    if isinstance(optimized, dict):
        found += sum(1 for k in _OPTIMIZED_KEYS if k in optimized)
    return round(found / total, 4)


def score_rouge_l(candidate: str, reference: str) -> float | None:
    """ROUGE-L F1 between model output and Phase 1 teacher reference."""
    if not candidate or not reference:
        return None
    try:
        from rouge_score import rouge_scorer as rs
        scorer = rs.RougeScorer(["rougeL"], use_stemmer=False)
        result = scorer.score(reference, candidate)
        return round(result["rougeL"].fmeasure, 4)
    except Exception as exc:
        log.warning("ROUGE-L failed", error=str(exc))
        return None


def score_bertscore(candidate: str, reference: str) -> float | None:
    """BERTScore F1 between model output and Phase 1 teacher reference.
    Uses distilbert-base-uncased — fast CPU inference, ~260MB download on first run.
    """
    if not candidate or not reference:
        return None
    try:
        import warnings
        from bert_score import score as bs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, F1 = bs(
                [candidate], [reference],
                model_type=_bertscore_model_type,
                lang="en",
                verbose=False,
            )
        return round(float(F1[0].item()), 4)
    except Exception as exc:
        log.warning("BERTScore failed", error=str(exc))
        return None


# ── LLM judge ──────────────────────────────────────────────────────────────

_last_judge_call: float = 0.0
_JUDGE_MIN_INTERVAL = 1.0  # DeepSeek has generous rate limits; 1s gap is sufficient


@api_cache(ttl=86400)
def call_llm_judge(prompt: str) -> str:
    """Judge via DeepSeek deepseek-chat (OpenAI-compatible). Cached 24h so reruns are instant."""
    global _last_judge_call
    import os
    from openai import OpenAI

    elapsed = time.monotonic() - _last_judge_call
    if elapsed < _JUDGE_MIN_INTERVAL:
        time.sleep(_JUDGE_MIN_INTERVAL - elapsed)

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=DEEPSEEK_BASE_URL + "/v1",
    )
    response = client.chat.completions.create(
        model=DEEPSEEK_AGENT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    _last_judge_call = time.monotonic()
    return response.choices[0].message.content


def parse_judge_score(raw: str) -> float | None:
    """Extract score from LLM judge JSON response."""
    try:
        return float(json.loads(raw).get("score", None))
    except Exception:
        pass
    m = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
    if m:
        try:
            return round(max(0.0, min(1.0, float(m.group(1)))), 4)
        except ValueError:
            pass
    return None


def parse_judge_passed(raw: str) -> bool:
    """Extract passed boolean from LLM judge JSON response."""
    try:
        val = json.loads(raw).get("passed")
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() == "true"
    except Exception:
        pass
    return "true" in raw.lower()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dig(d: dict, *keys):
    """Safe nested dict access."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _extract_activities(parsed: dict) -> str:
    """Pull activity text from any known output shape."""
    itinerary = (
        _dig(parsed, "optimized", "daily_itinerary")
        or parsed.get("daily_itinerary")
        or []
    )
    if isinstance(itinerary, list) and itinerary:
        parts = []
        for day in itinerary:
            if isinstance(day, dict):
                parts.append(str(day.get("activities", "")))
        return " ".join(p for p in parts if p)
    return ""
