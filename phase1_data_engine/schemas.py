"""
Pydantic v2 schemas for Phase 1 synthetic data.
All generation and validation code imports from here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

# ─── Canonical value sets ────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BUDGET_TIERS, TRIP_TYPES, INTENTS, ALL_CITIES

BUDGET_TIER_NAMES = list(BUDGET_TIERS.keys())
TripType    = Literal["Solo", "Family", "Group", "Couple", "Business"]
BudgetTier  = Literal["Shoestring", "Budget+", "Mid-Range", "Premium", "Ultra-Luxury"]


# ─── Persona ─────────────────────────────────────────────────────────────────

class PartySize(BaseModel):
    adults:   int = Field(ge=1, le=8)
    children: int = Field(ge=0, le=4)

    @property
    def total(self) -> int:
        return self.adults + self.children


class Persona(BaseModel):
    id:               int
    starting_city:    str
    destination_city: str
    type:             TripType
    size:             PartySize
    intents:          list[str] = Field(min_length=1, max_length=4)
    budget:           BudgetTier
    duration_days:    int = Field(ge=2, le=14)
    duration_nights:  int

    @field_validator("starting_city", "destination_city")
    @classmethod
    def must_be_known_city(cls, v: str) -> str:
        if v not in ALL_CITIES:
            raise ValueError(f"Unknown city: {v!r}")
        return v

    @field_validator("intents")
    @classmethod
    def intents_must_be_valid(cls, v: list[str]) -> list[str]:
        for intent in v:
            if intent not in INTENTS:
                raise ValueError(f"Unknown intent: {intent!r}")
        return v

    @model_validator(mode="after")
    def cities_must_differ(self) -> "Persona":
        if self.starting_city == self.destination_city:
            raise ValueError("starting_city and destination_city must differ")
        return self

    @model_validator(mode="after")
    def nights_consistent(self) -> "Persona":
        if self.duration_nights != self.duration_days - 1:
            raise ValueError("duration_nights must equal duration_days - 1")
        return self


# ─── Itinerary ───────────────────────────────────────────────────────────────

class DayPlan(BaseModel):
    day:          int = Field(ge=1)
    location:     str
    transit:      str
    stay_district: str
    activities:   str


class ItineraryVersion(BaseModel):
    total_trip_cost: float = Field(gt=0)
    daily_itinerary: list[DayPlan] = Field(min_length=1)


class ItineraryPair(BaseModel):
    baseline:       ItineraryVersion
    optimized:      ItineraryVersion
    pivot_analysis: str = Field(min_length=20)


# ─── Validated output record ─────────────────────────────────────────────────

class ValidationResult(BaseModel):
    passed:            bool
    fail_reason:       str | None = None
    savings_pct:       float | None = None
    expected_cost_min: float | None = None
    expected_cost_max: float | None = None


class SyntheticRecord(BaseModel):
    id:           str = Field(default_factory=lambda: str(uuid4()))
    phase:        int = 2
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    persona:    Persona
    pair:       ItineraryPair
    validation: ValidationResult