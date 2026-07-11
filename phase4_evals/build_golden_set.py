"""
Phase 4 — Build stratified golden eval set.
Samples 100 records from Phase 1 synthetic data:
  5 budget tiers × 5 trip types × 4 records per cell = 100 total.

Usage:
    python phase4_evals/build_golden_set.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SYNTHETIC_DIR, EVALS_DIR, BUDGET_TIERS, TRIP_TYPES
from utils.logger import get_logger

log = get_logger("phase4", "build_golden_set")

SYNTHETIC_FILE = SYNTHETIC_DIR / "v2_20260608_085742.jsonl"
OUTPUT_FILE = EVALS_DIR / "golden_set.jsonl"
RECORDS_PER_CELL = 4

# Phase 1 synthetic data uses these trip types
_TRIP_TYPES = TRIP_TYPES


def main() -> None:
    if not SYNTHETIC_FILE.exists():
        log.error("Synthetic file not found", path=str(SYNTHETIC_FILE))
        raise FileNotFoundError(f"Missing: {SYNTHETIC_FILE}")

    EVALS_DIR.mkdir(parents=True, exist_ok=True)

    # Load + filter
    all_records: list[dict] = []
    skipped = 0
    for line in SYNTHETIC_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not rec.get("validation", {}).get("passed", False):
            skipped += 1
            continue
        all_records.append(rec)

    log.info("Loaded records", total=len(all_records) + skipped, valid=len(all_records), skipped=skipped)

    # Bucket by (budget, trip_type)
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    uncategorized = 0
    for rec in all_records:
        persona = rec.get("persona", {})
        budget = persona.get("budget", "")
        trip_type = persona.get("type", "")
        if budget in BUDGET_TIERS and trip_type in _TRIP_TYPES:
            cells[(budget, trip_type)].append(rec)
        else:
            uncategorized += 1

    if uncategorized:
        log.warning("Records with unknown budget/type", count=uncategorized)

    # Stratified sampling: sort by savings_pct desc, take top RECORDS_PER_CELL
    golden: list[dict] = []
    print(f"\n{'Budget':<16} {'TripType':<14} {'Available':>9} {'Sampled':>8}")
    print("-" * 52)

    for budget in BUDGET_TIERS:
        for trip_type in _TRIP_TYPES:
            cell = cells[(budget, trip_type)]
            # Sort by savings_pct descending (best pivots first)
            cell.sort(
                key=lambda r: float(r.get("validation", {}).get("savings_pct", 0) or 0),
                reverse=True,
            )
            sampled = cell[:RECORDS_PER_CELL]
            golden.extend(sampled)
            n_sampled = len(sampled)
            flag = "" if n_sampled >= RECORDS_PER_CELL else " ⚠ (sparse)"
            print(f"{budget:<16} {trip_type:<14} {len(cell):>9} {n_sampled:>8}{flag}")

    print("-" * 52)
    print(f"{'TOTAL':<31} {len(all_records):>9} {len(golden):>8}\n")

    if len(golden) < 50:
        log.warning("Fewer than 50 golden records sampled — eval coverage will be limited", count=len(golden))

    # Write
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in golden:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info("Golden set written", records=len(golden), path=str(OUTPUT_FILE))
    print(f"Written: {OUTPUT_FILE} ({len(golden)} records)")


if __name__ == "__main__":
    main()
