"""
Phase 1 — Synthetic data generator (v2).
Generates baseline+optimized itinerary pairs from persona seeds.

Supports two providers selectable via --provider:
  groq   — Llama 3.3 70B via Groq API (14,400 req/day free, recommended)
  gemini — Gemini 2.0 Flash via Google AI Studio (regional free tier varies)

Usage:
    python phase1_data_engine/generate.py --provider groq --limit 5000 --batch-size 5
    python phase1_data_engine/generate.py --provider gemini --limit 5000 --batch-size 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from google import genai
from google.genai import types
from groq import Groq
from openai import OpenAI
from pydantic import ValidationError
from tqdm.asyncio import tqdm as atqdm

from config import (
    BUDGET_TIERS, HUB_CITIES, ALL_CITIES,
    TRIP_TYPES, NON_BUSINESS_INTENTS, SEEDS_DIR, SYNTHETIC_DIR,
    GEMINI_MODEL, GROQ_MODEL, OPENAI_MODEL,
)
from phase1_data_engine.schemas import (
    ItineraryPair, Persona, PartySize, SyntheticRecord
)
from phase1_data_engine.validate import validate
from utils.logger import get_logger

log_gen  = get_logger("phase1", "generation")
log_fail = get_logger("phase1", "validation_fails")
log_api  = get_logger("phase1", "api_errors")
log_prog = get_logger("phase1", "progress")

# ─── Seed Generation ─────────────────────────────────────────────────────────

def generate_seeds(num_seeds: int = 50_000) -> list[dict]:
    """Generate persona seeds with 5-tier budget system and origin+destination pairs."""
    random.seed(42)
    seeds = []
    for i in range(num_seeds):
        trip_type = random.choice(TRIP_TYPES)
        start     = random.choice(HUB_CITIES)
        dest      = random.choice([c for c in ALL_CITIES if c != start])

        if trip_type == "Business":
            intents = ["Business"] + random.sample(NON_BUSINESS_INTENTS, 1)
            size    = PartySize(adults=random.randint(1, 2), children=0)
            # Business travelers are never Shoestring/Budget+ — contradictory
            budget  = random.choice(["Mid-Range", "Premium", "Ultra-Luxury"])
        elif trip_type == "Family":
            intents = random.sample(NON_BUSINESS_INTENTS, random.randint(2, 3))
            size    = PartySize(adults=2, children=random.randint(1, 3))
        elif trip_type == "Solo":
            intents = random.sample(NON_BUSINESS_INTENTS, random.randint(1, 2))
            size    = PartySize(adults=1, children=0)
        elif trip_type == "Couple":
            intents = random.sample(NON_BUSINESS_INTENTS, random.randint(2, 3))
            size    = PartySize(adults=2, children=0)
        else:  # Group
            intents = random.sample(NON_BUSINESS_INTENTS, random.randint(2, 3))
            size    = PartySize(adults=random.randint(3, 8), children=0)

        if trip_type != "Business":  # Business budget already set above
            budget = random.choice(list(BUDGET_TIERS.keys()))
        duration = random.randint(3, 10)

        seeds.append({
            "id":               i,
            "starting_city":    start,
            "destination_city": dest,
            "type":             trip_type,
            "size":             size.model_dump(),
            "intents":          intents,
            "budget":           budget,
            "duration_days":    duration,
            "duration_nights":  duration - 1,
        })
    return seeds


def save_seeds(seeds: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s) + "\n")


# ─── Prompt Builder ──────────────────────────────────────────────────────────

def _build_prompt(batch: list[dict]) -> str:
    """
    Injects pre-computed cost bounds for every persona into the prompt.
    This is the key fix over v1 — the LLM has exact numbers, not just a tier name.
    """
    personas_with_bounds = []
    for p in batch:
        tier    = BUDGET_TIERS[p["budget"]]
        people  = p["size"]["adults"] + p["size"]["children"]
        days    = p["duration_days"]
        lo      = round(tier["min_daily"] * people * days * 0.8)
        hi      = round(tier["max_daily"] * people * days * 1.2)
        min_sav = tier["min_savings_pct"]
        personas_with_bounds.append({**p, "_cost_bounds": {"min": lo, "max": hi, "min_savings_pct": min_sav}})

    personas_json = json.dumps(personas_with_bounds, indent=2, ensure_ascii=False)

    # For batch size 1, use a simpler single-object schema (far fewer structure failures)
    if len(batch) == 1:
        p = personas_with_bounds[0]
        return f"""You are ItinerAI-Bench, an Indian travel planner. Generate a baseline and optimized travel plan.

PERSONA:
{json.dumps(p, indent=2, ensure_ascii=False)}

HARD RULES:
- baseline.total_trip_cost must be between {p['_cost_bounds']['min']} and {p['_cost_bounds']['max']} INR.
- optimized.total_trip_cost must be at least {p['_cost_bounds']['min_savings_pct'] + 1:.0f}% cheaper than baseline. Be generous — aim for 8-15% savings.
- daily_itinerary must have exactly {p['duration_days']} entries (day 1 = depart, last day = return).
- {"NEVER use hostels or dorms — this is a " + p['type'] + "/" + p['budget'] + " traveler." if p['type'] in ('Business','Premium','Ultra-Luxury') or p['budget'] in ('Premium','Ultra-Luxury') else "Shoestring/Budget+: shift hotel to guesthouse or hostel in optimized plan."}
- Optimized: shift stay to a cheaper nearby district (Area Pivot) and use cheaper transit (Transit Pivot).

Return a JSON object (not an array) with this exact structure:
{{
  "id": {p['id']},
  "baseline": {{
    "total_trip_cost": <INR number>,
    "daily_itinerary": [{{"day": 1, "location": "...", "transit": "...", "stay_district": "...", "activities": "..."}}]
  }},
  "optimized": {{
    "total_trip_cost": <INR number, must be {p['_cost_bounds']['min_savings_pct'] + 1:.0f}%+ cheaper>,
    "daily_itinerary": [{{"day": 1, "location": "...", "transit": "...", "stay_district": "...", "activities": "..."}}]
  }},
  "pivot_analysis": "Explain which Area/Transit/Accommodation pivots saved money."
}}"""

    # Batch size > 1: use array format
    return f"""You are ItinerAI-Bench, an Indian travel planner. Generate baseline and optimized travel plans.

RULES:
1. baseline.total_trip_cost MUST be within _cost_bounds.min and _cost_bounds.max (INR).
2. optimized must save at least (_cost_bounds.min_savings_pct + 1)% vs baseline. Aim for 8-15% savings.
3. daily_itinerary must have exactly duration_days entries.
4. NEVER use hostels for Business, Premium, or Ultra-Luxury personas.
5. Optimized: Area Pivot (cheaper nearby district) + Transit Pivot (train/bus over direct flight for budget tiers).

PERSONAS:
{personas_json}

Return JSON object with key "itineraries" = list of exactly {len(batch)} objects:
{{"itineraries": [{{"id": <int>, "baseline": {{"total_trip_cost": <num>, "daily_itinerary": [...]}}, "optimized": {{"total_trip_cost": <num>, "daily_itinerary": [...]}}, "pivot_analysis": "..."}}]}}"""


# ─── API Calls ───────────────────────────────────────────────────────────────

import re as _re

def _parse_retry_after(exc: Exception) -> float:
    """Extract 'retry after N seconds' from a 429 error message."""
    match = _re.search(r'retry[^\d]+(\d+(?:\.\d+)?)\s*s', str(exc), _re.IGNORECASE)
    return float(match.group(1)) + 2 if match else 65.0


def _unwrap_json(raw: dict | list) -> list:
    """Return the list of itineraries regardless of how the LLM wrapped it."""
    if isinstance(raw, list):
        return raw
    for key in ("itineraries", "results", "data"):
        if key in raw and isinstance(raw[key], list):
            return raw[key]
    # Single-object response (batch_size=1): has "baseline" and "optimized" directly
    if isinstance(raw, dict) and "baseline" in raw and "optimized" in raw:
        return [raw]
    return []


async def _call_provider(
    client,
    provider: str,
    batch: list[dict],
    batch_idx: int,
    sem: asyncio.Semaphore,
    max_retries: int = 5,
) -> list[dict]:
    """Unified async call that works for both 'groq' and 'gemini' providers."""
    prompt = _build_prompt(batch)

    async with sem:
        for attempt in range(max_retries):
            try:
                if provider in ("openai", "groq"):
                    model = OPENAI_MODEL if provider == "openai" else GROQ_MODEL
                    response = await asyncio.to_thread(
                        client.chat.completions.create,
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a strict JSON data generator. Output only valid JSON, no markdown."},
                            {"role": "user",   "content": prompt},
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.7,
                    )
                    raw = json.loads(response.choices[0].message.content)

                else:  # gemini
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model=GEMINI_MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.7,
                        ),
                    )
                    raw = json.loads(response.text)

                result = _unwrap_json(raw)
                if not result:
                    log_api.warning("Unexpected JSON shape", batch_idx=batch_idx, provider=provider)
                return result

            except Exception as exc:
                err_str = str(exc)
                is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower() or "RESOURCE_EXHAUSTED" in err_str
                is_tpd_exhausted = "tokens per day" in err_str.lower() and "429" in err_str

                if is_tpd_exhausted:
                    # Daily token budget gone — no point retrying, stop the run
                    log_api.error(
                        "Daily token budget (TPD) exhausted — stopping. Resume tomorrow.",
                        batch_idx=batch_idx,
                        error=err_str[:200],
                    )
                    raise RuntimeError("TPD_EXHAUSTED")

                if attempt == max_retries - 1:
                    log_api.error("Batch permanently failed", batch_idx=batch_idx, error=err_str[:300])
                    return []

                wait = _parse_retry_after(exc) if is_rate_limit else 2 ** attempt
                log_api.warning("Retrying batch", batch_idx=batch_idx, attempt=attempt + 1, wait_s=wait, reason="rate_limit" if is_rate_limit else "error")
                await asyncio.sleep(wait)

    return []


# ─── Processing ──────────────────────────────────────────────────────────────

def _parse_and_validate(
    raw_item: dict,
    persona_dict: dict,
    output_path: Path,
) -> bool:
    """Parse one LLM result, validate, and append to output file if valid."""
    try:
        persona = Persona(**persona_dict)
        pair    = ItineraryPair(**{
            "baseline":       raw_item["baseline"],
            "optimized":      raw_item["optimized"],
            "pivot_analysis": raw_item.get("pivot_analysis", ""),
        })
    except (ValidationError, KeyError, TypeError) as exc:
        log_fail.info(
            "Structure validation failed",
            persona_id=persona_dict.get("id"),
            error=str(exc)[:200],
        )
        return False

    result = validate(persona, pair)
    if not result.passed:
        log_fail.info(
            "Logic validation failed",
            persona_id=persona.id,
            fail_reason=result.fail_reason,
            baseline_cost=pair.baseline.total_trip_cost,
            budget=persona.budget,
        )
        return False

    record = SyntheticRecord(persona=persona, pair=pair, validation=result)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
    return True


async def run(
    seeds: list[dict],
    output_path: Path,
    batch_size: int = 5,
    concurrency: int = 8,
    provider: str = "groq",
) -> dict:
    """Main async loop. Returns summary stats dict."""
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set. Add it to .env")
        client = OpenAI(api_key=api_key)
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set. Add it to .env")
        client = Groq(api_key=api_key)
    else:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set. Add it to .env")
        client = genai.Client(api_key=api_key)

    # Checkpoint: skip already-done IDs
    done_ids: set[int] = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["persona"]["id"])
                except Exception:
                    pass

    remaining  = [s for s in seeds if s["id"] not in done_ids]
    batches    = [remaining[i:i+batch_size] for i in range(0, len(remaining), batch_size)]
    sem        = asyncio.Semaphore(concurrency)

    total_valid = sum(1 for _ in open(output_path, encoding="utf-8")) if output_path.exists() else 0
    total_processed = len(done_ids)
    total_failed    = 0
    start_time      = time.time()

    log_prog.info(
        "Generation started",
        total_seeds=len(seeds),
        already_done=len(done_ids),
        remaining=len(remaining),
        batches=len(batches),
    )

    id_to_persona = {s["id"]: s for s in remaining}

    async def process_batch(batch: list[dict], batch_idx: int, client) -> None:
        nonlocal total_valid, total_processed, total_failed

        results = await _call_provider(client, provider, batch, batch_idx, sem)
        batch_map = {b["id"]: b for b in batch}

        log_gen.info(
            "Batch complete",
            batch_idx=batch_idx,
            returned=len(results),
            expected=len(batch),
        )

        for item in results:
            p_id = item.get("id")
            if p_id not in batch_map:
                continue
            ok = _parse_and_validate(item, batch_map[p_id], output_path)
            if ok:
                total_valid += 1
            else:
                total_failed += 1
            total_processed += 1

        # Progress snapshot every 50 batches
        if batch_idx % 50 == 0:
            elapsed   = time.time() - start_time
            rate      = total_processed / max(elapsed, 1)
            remaining_count = len(seeds) - len(done_ids) - total_processed
            eta_min   = (remaining_count / max(rate, 0.01)) / 60
            pass_rate = total_valid / max(total_processed, 1) * 100
            log_prog.info(
                "Progress snapshot",
                processed=total_processed,
                valid=total_valid,
                failed=total_failed,
                pass_rate_pct=round(pass_rate, 1),
                eta_minutes=round(eta_min, 1),
            )

    tasks = [process_batch(b, i, client) for i, b in enumerate(batches)]
    try:
        for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating"):
            await coro
    except RuntimeError as e:
        if "TPD_EXHAUSTED" in str(e):
            print("\n⚠️  Daily token limit reached. Progress saved via checkpoint.")
            print("   Resume tomorrow (after 5:30 AM IST) with the same command.")
        else:
            raise

    elapsed   = time.time() - start_time
    pass_rate = total_valid / max(total_processed, 1) * 100
    summary   = {
        "total_seeds":    len(seeds),
        "processed":      total_processed + len(done_ids),
        "valid":          total_valid,
        "failed":         total_failed,
        "pass_rate_pct":  round(pass_rate, 1),
        "elapsed_min":    round(elapsed / 60, 1),
        "output_file":    str(output_path),
    }
    log_prog.info("Generation complete", **summary)
    return summary


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ItinerAI-Bench Phase 1 — synthetic data generation")
    parser.add_argument("--limit",      type=int, default=50_000, help="Max seeds to process")
    parser.add_argument("--batch-size", type=int, default=5,      help="Personas per API call")
    parser.add_argument("--concurrency",type=int, default=2,
                        help="Concurrent API workers. Groq free tier = 30 RPM. "
                             "At batch=1 each call takes ~4s, so concurrency=2 → ~30 RPM. "
                             "At batch=5 each call takes ~8s, so concurrency=4 → ~30 RPM.")
    parser.add_argument("--output",     type=str, default=None,   help="Output JSONL path")
    parser.add_argument("--regen-seeds", action="store_true",     help="Regenerate seed file first")
    parser.add_argument("--provider",   type=str, default="openai", choices=["openai", "groq", "gemini"], help="LLM provider (default: openai)")
    args = parser.parse_args()

    seeds_path = SEEDS_DIR / "persona_seeds_50k.jsonl"

    # Regenerate or load seeds
    if args.regen_seeds or not seeds_path.exists():
        print("Generating 50k persona seeds...")
        seeds = generate_seeds(50_000)
        save_seeds(seeds, seeds_path)
        print(f"Saved {len(seeds)} seeds → {seeds_path}")
    else:
        with open(seeds_path, encoding="utf-8") as f:
            seeds = [json.loads(line) for line in f]
        print(f"Loaded {len(seeds)} seeds from {seeds_path}")

    seeds = seeds[: args.limit]

    ts          = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else SYNTHETIC_DIR / f"v2_{ts}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Output → {output_path}")
    print(f"Provider: {args.provider}")
    summary = asyncio.run(run(seeds, output_path, args.batch_size, args.concurrency, args.provider))

    print("\n─── Summary ───────────────────────────────")
    for k, v in summary.items():
        print(f"  {k:<20} {v}")


if __name__ == "__main__":
    main()