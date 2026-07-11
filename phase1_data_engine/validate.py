"""
Phase 1 validation logic.
Three gates, in order:
  1. Vibe Check   — persona type vs accommodation type
  2. Savings Gate — optimized cost ≥ min_savings_pct cheaper than baseline
  3. Budget Bounds — baseline cost within ±20% of tier × people × days
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BUDGET_TIERS, NO_HOSTEL_TYPES, VALIDATION_COST_MARGIN
from phase1_data_engine.schemas import ItineraryPair, Persona, ValidationResult


def validate(persona: Persona, pair: ItineraryPair) -> ValidationResult:
    """Returns a ValidationResult. Call result.passed to check outcome."""

    # ── Gate 1: Vibe Check ───────────────────────────────────────────────────
    if persona.type in NO_HOSTEL_TYPES or persona.budget in NO_HOSTEL_TYPES:
        itinerary_text = (
            str([d.model_dump() for d in pair.baseline.daily_itinerary]).lower()
            + str([d.model_dump() for d in pair.optimized.daily_itinerary]).lower()
        )
        hostel_words = {"hostel", "dormitory", "dorm room", "shared dorm"}
        for word in hostel_words:
            if word in itinerary_text:
                return ValidationResult(
                    passed=False,
                    fail_reason=f"Vibe: {persona.type}/{persona.budget} persona assigned hostel/dorm."
                )

    baseline_cost  = pair.baseline.total_trip_cost
    optimized_cost = pair.optimized.total_trip_cost

    # ── Gate 2: Savings Validation ────────────────────────────────────────────
    tier_cfg     = BUDGET_TIERS[persona.budget]
    min_savings  = tier_cfg["min_savings_pct"] / 100.0

    if optimized_cost >= baseline_cost:
        return ValidationResult(
            passed=False,
            fail_reason=f"Savings: optimized (₹{optimized_cost:,.0f}) ≥ baseline (₹{baseline_cost:,.0f})."
        )

    savings_pct = (baseline_cost - optimized_cost) / baseline_cost
    if savings_pct < (min_savings - 0.001):  # 0.001 tolerance for floating point
        return ValidationResult(
            passed=False,
            fail_reason=(
                f"Savings: {savings_pct*100:.1f}% < required {min_savings*100:.1f}% "
                f"for {persona.budget} tier."
            )
        )

    # ── Gate 3: Budget Bounds ─────────────────────────────────────────────────
    total_people  = persona.size.total
    days          = persona.duration_days
    expected_min  = tier_cfg["min_daily"] * total_people * days * (1 - VALIDATION_COST_MARGIN)
    expected_max  = tier_cfg["max_daily"] * total_people * days * (1 + VALIDATION_COST_MARGIN)

    if baseline_cost < expected_min or baseline_cost > expected_max:
        return ValidationResult(
            passed=False,
            fail_reason=(
                f"Budget bounds: ₹{baseline_cost:,.0f} outside "
                f"[₹{expected_min:,.0f}, ₹{expected_max:,.0f}] "
                f"({persona.budget}, {total_people} people, {days} days)."
            ),
            savings_pct=round(savings_pct * 100, 2),
            expected_cost_min=round(expected_min, 0),
            expected_cost_max=round(expected_max, 0),
        )

    return ValidationResult(
        passed=True,
        savings_pct=round(savings_pct * 100, 2),
        expected_cost_min=round(expected_min, 0),
        expected_cost_max=round(expected_max, 0),
    )