"""
Phase 4 — Step 1 of 2: generate model responses.
Runs Ollama inference only — no metrics, no API calls.
Saves raw outputs to data/evals/responses_<timestamp>.jsonl.

Run this once (~23 hrs on MacBook Air CPU). Then run score_responses.py
as many times as needed to recompute metrics without re-running Ollama.

Usage:
    python phase4_evals/generate_responses.py
    python phase4_evals/generate_responses.py --model pivotai-ft
    python phase4_evals/generate_responses.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EVALS_DIR, SLM_FT_MODEL, SLM_DIST_MODEL, SLM_CURRICULUM_MODEL, SLM_BASELINE_MODEL
from utils.logger import get_logger
from phase4_evals.utils import build_prompt, call_ollama

log = get_logger("phase4", "generate_responses")

ALL_MODELS = [SLM_FT_MODEL, SLM_DIST_MODEL, SLM_CURRICULUM_MODEL, SLM_BASELINE_MODEL]
GOLDEN_SET_PATH = EVALS_DIR / "golden_set.jsonl"


def _get_done_keys(evals_dir: Path) -> set[tuple[str, str]]:
    """Return (model_name, golden_record_id) pairs already saved without errors."""
    done: set[tuple[str, str]] = set()
    for f in evals_dir.glob("responses_*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if not r.get("raw_output", "").startswith("ERROR:"):
                    done.add((r["model_name"], r["golden_record_id"]))
            except Exception:
                continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pivotai model responses")
    parser.add_argument("--model", default=None, help="Run one model only")
    parser.add_argument("--limit", type=int, default=None, help="Limit golden records")
    args = parser.parse_args()

    if not GOLDEN_SET_PATH.exists():
        print("ERROR: Golden set not found. Run: python phase4_evals/build_golden_set.py")
        sys.exit(1)

    golden_records = []
    for line in GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                golden_records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if args.limit:
        golden_records = golden_records[:args.limit]

    models = [args.model] if args.model else ALL_MODELS
    EVALS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = EVALS_DIR / f"responses_{timestamp}.jsonl"

    done = _get_done_keys(EVALS_DIR)
    total = len(models) * len(golden_records)
    skipped = sum(
        1 for m in models for r in golden_records
        if (m, r.get("id") or r.get("record_id", "")) in done
    )

    print(f"Golden records : {len(golden_records)}")
    print(f"Models         : {models}")
    print(f"Already done   : {skipped}/{total} (will skip)")
    print(f"Output         : {output_path}\n")

    success = error = 0
    with tqdm(total=total, desc="Generating", unit="response") as pbar:
        for model_name in models:
            for golden in golden_records:
                record_id = golden.get("id") or golden.get("record_id", "")
                if not record_id:
                    record_id = json.dumps(golden.get("persona", {}), sort_keys=True)[:32]

                if (model_name, record_id) in done:
                    pbar.update(1)
                    continue

                persona = golden.get("persona", {})
                prompt = build_prompt(persona, model_name)

                try:
                    raw = call_ollama(model_name, prompt)
                    success += 1
                except Exception as exc:
                    log.warning("Ollama call failed", model=model_name, error=str(exc))
                    raw = f"ERROR: {exc}"
                    error += 1

                record = {
                    "model_name": model_name,
                    "golden_record_id": record_id,
                    "persona": persona,
                    "prompt_used": prompt,
                    "raw_output": raw,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                pbar.set_description(f"Generating [{model_name}] ok={success} err={error}")
                pbar.update(1)

    print(f"\n✓ Done: {success} generated, {error} errors")
    print(f"  Output: {output_path}")
    print(f"\nNext: python phase4_evals/score_responses.py")
    log.info("Generation complete", success=success, error=error, output=str(output_path))


if __name__ == "__main__":
    main()
