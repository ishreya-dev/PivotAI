# Phase 1 — Synthetic Data Engine

Generates ground-truth training data: validated `(baseline, optimized)` itinerary pairs across 20 Indian cities, 5 budget tiers, 5 trip types, and 8 intent categories.

## Final Output

**Canonical file**: `data/synthetic/v2_20260608_085742.jsonl`

| Metric | Value |
|--------|-------|
| Records | 5,000 |
| Pass rate | 100% |
| Avg savings | 9.0% |
| Savings range | 7% – 20% |
| Budget tiers | All 5 (Shoestring → Ultra-Luxury) |
| Trip types | All 5 (Solo / Couple / Family / Group / Business) |
| Intents | All 8 (Adventure / Relax / Cultural / Business / Foodie / Nightlife / Shopping / Wildlife) |
| Cities | 20 Indian cities (8 hubs + 12 leisure) |
| Duration range | 3 – 10 days |
| Teacher model | OpenAI gpt-4o-mini |

---

## Record Schema

```json
{
  "id": "uuid4",
  "generated_at": "ISO-8601",
  "persona": {
    "starting_city": "Delhi",
    "destination_city": "Goa",
    "type": "Solo",
    "size": { "adults": 1, "children": 0 },
    "intents": ["Relax", "Foodie"],
    "budget": "Budget+",
    "duration_days": 5,
    "duration_nights": 4
  },
  "pair": {
    "baseline":  { "total_trip_cost": 18000, "daily_itinerary": [ { "day", "location", "transit", "stay_district", "activities" } ] },
    "optimized": { "total_trip_cost": 15500, "daily_itinerary": [ ... ] },
    "pivot_analysis": "Switched from IndiGo flight to Sleeper AC train, saving ₹2,100 per person..."
  },
  "validation": {
    "passed": true,
    "savings_pct": 13.9,
    "expected_cost_min": 12000,
    "expected_cost_max": 36000
  }
}
```

---

## Files

| File | Purpose |
|------|---------|
| `schemas.py` | Pydantic v2 models for all data structures |
| `validate.py` | 3-gate validator (hostel check, savings gate, budget bounds) |
| `generate.py` | Async gpt-4o-mini pipeline with checkpoint resume |

---

## Validation Logic

Three gates applied in order. Any failure → record discarded and logged to `logs/phase1/validation_fails.log`.

1. **Hostel check**: Business / Premium / Ultra-Luxury personas must never be assigned hostels or dorms.
2. **Savings gate**: `(baseline - optimized) / baseline ≥ tier.min_savings_pct`
   - Shoestring / Budget+: ≥ 4%
   - Mid-Range / Premium / Ultra-Luxury: ≥ 5%
3. **Budget bounds**: `baseline_cost ∈ [tier_min × people × days × 0.8, tier_max × people × days × 1.2]`

---

## How to Re-run or Extend

The canonical output file already exists. Only re-run if you need more records.

```bash
# Checkpoint-safe — auto-skips already-generated records
python phase1_data_engine/generate.py \
  --provider openai \
  --limit 5000 \
  --batch-size 1 \
  --concurrency 10

# Check record count
wc -l data/synthetic/v2_20260608_085742.jsonl
```

---

## Key Design Decisions

**Why gpt-4o-mini?** Reliable structured JSON output, fast, and cost $4 for 5,000 records — exactly matching the Phase 2 DeepSeek budget to keep the fine-tune vs. distill comparison methodologically fair. OpenAI was the only provider that consistently hit <5% rejection rate on structured output without extra prompting.

**Why inject cost bounds into the prompt?** v1 only gave the tier name — the LLM inferred the INR range itself, causing 60%+ failures. Injecting `min_daily` and `max_daily` from `config.BUDGET_TIERS` directly into the prompt dropped failures to near zero.

**Why 5 budget tiers?** `hotel_stars` (1–5) maps directly to Overpass OSM hotel quality filters and `price_level` (0–4) maps to OSM restaurant filters. Phase 2 agents use `config.BUDGET_TIERS[tier]["hotel_stars"]` directly without any translation layer.

---

## Log Files

| File | Contents |
|------|---------|
| `logs/phase1/generation.log` | Per-batch status, retry counts |
| `logs/phase1/validation_fails.log` | Rejected records with reason and cost details |
| `logs/phase1/api_errors.log` | API errors (rate limit, timeout, JSON parse) |
| `logs/phase1/progress.log` | Running summary: processed, valid, failed, pass rate |