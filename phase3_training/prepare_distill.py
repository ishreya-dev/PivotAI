"""
Phase 3 — Distillation data preparation.
Converts Phase 2 agent traces into Alpaca instruction format for knowledge distillation.

Reads from the single canonical trace file: data/traces/agent_traces_all.jsonl
(already filtered to 500 quality records — deduped by persona, api_calls >= 50)

Output:
  data/training/distill_train.jsonl  (~450 records)
  data/training/distill_val.jsonl    (~50 records)

Run:
  python phase3_training/prepare_distill.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRACES_DIR, TRAINING_DIR
from utils.logger import get_logger

log = get_logger("phase3", "prepare_distill")

_INSTRUCTION = (
    "Act as pivotai Supervisor for an Indian domestic trip. Coordinate the Analyst, "
    "Concierge, and Optimizer agents to find Price-Pivot Points and produce an optimized "
    "itinerary. Show the reasoning chain for each agent handoff, then provide the final "
    "pivot analysis and optimized itinerary."
)


def build_reasoning_output(record: dict) -> str:
    parts = []
    for step in record.get("agent_steps", []):
        agent = step["agent_name"].capitalize()
        reasoning = (step.get("reasoning") or "").strip()
        if reasoning:
            parts.append(f"[{agent}]\n{reasoning}")

    pivot = (record.get("pivot_analysis") or "").strip()
    if pivot:
        parts.append(f"[PIVOT ANALYSIS]\n{pivot}")

    optimized = record.get("agent_optimized")
    if optimized:
        parts.append(f"[OPTIMIZED ITINERARY]\n{json.dumps(optimized, ensure_ascii=False)}")

    return "\n\n".join(parts)


def build_alpaca_distill(record: dict) -> dict:
    return {
        "instruction": _INSTRUCTION,
        "input": json.dumps(record["persona"], ensure_ascii=False),
        "output": build_reasoning_output(record),
    }


def load_traces() -> list[dict]:
    canonical = TRACES_DIR / "agent_traces_all.jsonl"
    if not canonical.exists():
        raise FileNotFoundError(f"{canonical} not found. Expected the merged canonical trace file.")

    raw = []
    for line in canonical.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return raw


def filter_and_dedup(raw: list[dict]) -> list[dict]:
    # Canonical file is already deduped by persona and filtered to api_calls >= 50.
    # Just apply basic structural quality check.
    final = [
        r for r in raw
        if r.get("agent_optimized") and len(r.get("agent_steps", [])) >= 2
    ]
    log.info("Quality check", total=len(raw), kept=len(final))
    return final


def split_and_write(
    records: list[dict],
    train_path: Path,
    val_path: Path,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[int, int]:
    random.seed(seed)
    shuffled = records[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - val_ratio))
    train, val = shuffled[:split], shuffled[split:]

    alpaca_train = [build_alpaca_distill(r) for r in train]
    alpaca_val = [build_alpaca_distill(r) for r in val]

    train_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in alpaca_train))
    val_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in alpaca_val))
    return len(alpaca_train), len(alpaca_val)


def main() -> None:
    log.info("Loading Phase 2 traces", dir=str(TRACES_DIR))
    raw = load_traces()
    log.info("Raw unique traces loaded", count=len(raw))

    filtered = filter_and_dedup(raw)

    if len(filtered) < 400:
        log.warning("Fewer than 400 training-quality traces", count=len(filtered))

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    n_train, n_val = split_and_write(
        filtered,
        TRAINING_DIR / "distill_train.jsonl",
        TRAINING_DIR / "distill_val.jsonl",
    )

    log.info("Distillation data written", train=n_train, val=n_val)
    print(f"Canonical traces    : {len(raw):,}")
    print(f"After quality check : {len(filtered):,}")
    print(f"distill_train.jsonl : {n_train:,} records")
    print(f"distill_val.jsonl   : {n_val:,} records")


if __name__ == "__main__":
    main()
