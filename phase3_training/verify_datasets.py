"""
Phase 3 — Dataset verification.
Validates all six training JSONL files before Colab upload.

Checks per file:
  - File exists and is non-empty
  - Every line parses as valid JSON
  - Every record has 'instruction', 'input', 'output' keys
  - 'output' field is a non-empty string
  - Record count meets minimum threshold

Run:
  python phase3_training/verify_datasets.py
  echo $?   # 0 = all pass, 1 = any fail
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRAINING_DIR
from utils.logger import get_logger

log = get_logger("phase3", "verify")

_FILES = [
    ("ft_train.jsonl",          4000),
    ("ft_val.jsonl",            100),
    ("distill_train.jsonl",     350),
    ("distill_val.jsonl",       30),
    ("curriculum_stage1.jsonl", 4000),
    ("curriculum_stage2.jsonl", 350),
]


def verify_file(path: Path, min_records: int) -> bool:
    if not path.exists():
        print(f"  FAIL  {path.name} — file not found")
        log.error("File missing", file=path.name)
        return False

    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        print(f"  FAIL  {path.name} — file is empty")
        return False

    errors = []
    for i, line in enumerate(lines, 1):
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {i}: JSON parse error — {e}")
            continue

        for key in ("instruction", "input", "output"):
            if key not in r:
                errors.append(f"line {i}: missing '{key}'")
            elif not isinstance(r[key], str) or not r[key].strip():
                errors.append(f"line {i}: '{key}' is empty or not a string")

        if len(errors) >= 5:
            errors.append("(stopping after 5 errors)")
            break

    if errors:
        print(f"  FAIL  {path.name} — {len(errors)} error(s):")
        for e in errors[:5]:
            print(f"         {e}")
        return False

    if len(lines) < min_records:
        print(f"  FAIL  {path.name} — {len(lines):,} records, expected >= {min_records:,}")
        return False

    print(f"  PASS  {path.name} — {len(lines):,} records")
    return True


def main() -> None:
    print("Verifying Phase 3 training datasets...\n")
    all_pass = True

    for filename, min_count in _FILES:
        ok = verify_file(TRAINING_DIR / filename, min_count)
        all_pass = all_pass and ok

    print()
    if all_pass:
        print("All checks passed. Safe to upload to Colab.")
        log.info("All dataset checks passed")
        sys.exit(0)
    else:
        print("Some checks failed. Fix issues before uploading.")
        log.error("Dataset verification failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
