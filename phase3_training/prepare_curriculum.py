"""
Phase 3 — Curriculum training data preparation.
Copies ft_train and distill_train into stage-tagged files for sequential training.

Stage 1 (domain knowledge) → curriculum_stage1.jsonl  (copy of ft_train)
Stage 2 (reasoning)        → curriculum_stage2.jsonl  (copy of distill_train)

Must be run AFTER prepare_ft.py and prepare_distill.py.

Run:
  python phase3_training/prepare_curriculum.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRAINING_DIR
from utils.logger import get_logger

log = get_logger("phase3", "prepare_curriculum")


def tag_and_copy(src: Path, dst: Path, stage: int) -> int:
    """Copy src JSONL to dst, adding a 'stage' field to each record."""
    lines = [l.strip() for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    tagged = []
    for line in lines:
        r = json.loads(line)
        r["stage"] = stage
        tagged.append(json.dumps(r, ensure_ascii=False))
    dst.write_text("\n".join(tagged), encoding="utf-8")
    return len(tagged)


def main() -> None:
    ft_train = TRAINING_DIR / "ft_train.jsonl"
    distill_train = TRAINING_DIR / "distill_train.jsonl"

    if not ft_train.exists():
        raise FileNotFoundError(
            f"{ft_train} not found. Run 'python phase3_training/prepare_ft.py' first."
        )
    if not distill_train.exists():
        raise FileNotFoundError(
            f"{distill_train} not found. Run 'python phase3_training/prepare_distill.py' first."
        )

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    n1 = tag_and_copy(ft_train, TRAINING_DIR / "curriculum_stage1.jsonl", stage=1)
    n2 = tag_and_copy(distill_train, TRAINING_DIR / "curriculum_stage2.jsonl", stage=2)

    log.info("Curriculum data written", stage1=n1, stage2=n2)
    print(f"curriculum_stage1.jsonl : {n1:,} records  (Phase 1 domain knowledge)")
    print(f"curriculum_stage2.jsonl : {n2:,} records  (Phase 2 agent reasoning)")


if __name__ == "__main__":
    main()
