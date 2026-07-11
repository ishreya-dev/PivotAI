"""
Phase 3 — Fine-tuning data preparation.
Converts Phase 1 synthetic pairs into Alpaca instruction format for SFT.

Output:
  data/training/ft_train.jsonl  (~4,750 records)
  data/training/ft_val.jsonl    (~250 records)

Run:
  python phase3_training/prepare_ft.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SYNTHETIC_DIR, TRAINING_DIR
from utils.logger import get_logger

log = get_logger("phase3", "prepare_ft")

_INSTRUCTION = (
    "Act as pivotai Optimizer. Given a traveler persona for an Indian domestic trip, "
    "produce an optimized day-by-day itinerary that minimizes total cost while respecting "
    "the budget tier, trip type, and traveler intents. Identify the primary Price-Pivot Point "
    "(transit, accommodation, or activity substitution that saves ≥5%) and explain it clearly."
)


def build_alpaca_ft(record: dict) -> dict:
    pair = record["pair"]
    output = {
        "optimized": pair["optimized"],
        "pivot_analysis": pair["pivot_analysis"],
        "savings_pct": record["validation"]["savings_pct"],
    }
    return {
        "instruction": _INSTRUCTION,
        "input": json.dumps(record["persona"], ensure_ascii=False),
        "output": json.dumps(output, ensure_ascii=False),
    }


def split_and_write(
    records: list[dict],
    train_path: Path,
    val_path: Path,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[int, int]:
    random.seed(seed)
    shuffled = records[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - val_ratio))
    train, val = shuffled[:split], shuffled[split:]

    train_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train))
    val_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in val))
    return len(train), len(val)


def main() -> None:
    source = next(SYNTHETIC_DIR.glob("v2_*.jsonl"), None)
    if not source:
        raise FileNotFoundError(f"No Phase 1 JSONL found in {SYNTHETIC_DIR}")

    log.info("Loading Phase 1 data", file=source.name)
    records = []
    skipped = 0
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if not r.get("validation", {}).get("passed", False):
            skipped += 1
            continue
        records.append(build_alpaca_ft(r))

    log.info("Phase 1 loaded", total=len(records) + skipped, skipped=skipped, kept=len(records))

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    n_train, n_val = split_and_write(
        records,
        TRAINING_DIR / "ft_train.jsonl",
        TRAINING_DIR / "ft_val.jsonl",
    )

    log.info("Fine-tuning data written", train=n_train, val=n_val)
    print(f"ft_train.jsonl  : {n_train:,} records")
    print(f"ft_val.jsonl    : {n_val:,} records")


if __name__ == "__main__":
    main()
