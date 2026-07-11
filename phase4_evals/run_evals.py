"""
Phase 4 — Convenience wrapper: generate responses then score them.

Prefer running the two steps separately for better control:
    python phase4_evals/generate_responses.py   # Step 1: Ollama only (~23 hrs, run once)
    python phase4_evals/score_responses.py      # Step 2: all metrics (fast, rerun anytime)

This wrapper runs both steps sequentially. Useful for smoke tests with --limit.

Usage:
    python phase4_evals/run_evals.py --limit 5 --no-judge   # fast smoke test
    python phase4_evals/run_evals.py --model pivotai-ft    # one model, both steps
    python phase4_evals/run_evals.py                        # full run (both steps)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pivotai eval suite — runs generate_responses then score_responses"
    )
    parser.add_argument("--model", default=None, help="Run one model only")
    parser.add_argument("--limit", type=int, default=None, help="Limit golden records per model")
    parser.add_argument("--no-judge", action="store_true", help="Skip DeepSeek judge (local metrics only)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    python = sys.executable

    # ── Step 1: generate responses ─────────────────────────────────────────────
    gen_cmd = [python, "phase4_evals/generate_responses.py"]
    if args.model:
        gen_cmd += ["--model", args.model]
    if args.limit:
        gen_cmd += ["--limit", str(args.limit)]

    print("=" * 60)
    print("STEP 1: Generating model responses (Ollama inference)")
    print("=" * 60)
    result = subprocess.run(gen_cmd, cwd=project_root)
    if result.returncode != 0:
        print(f"\nERROR: generate_responses.py exited with code {result.returncode}")
        sys.exit(result.returncode)

    # ── Step 2: score responses ────────────────────────────────────────────────
    score_cmd = [python, "phase4_evals/score_responses.py"]
    if args.no_judge:
        score_cmd += ["--no-judge"]

    print("\n" + "=" * 60)
    print("STEP 2: Scoring responses (metrics + DeepSeek judge)")
    print("=" * 60)
    result = subprocess.run(score_cmd, cwd=project_root)
    if result.returncode != 0:
        print(f"\nERROR: score_responses.py exited with code {result.returncode}")
        sys.exit(result.returncode)

    print("\n" + "=" * 60)
    print("Both steps complete. Run: python phase4_evals/compare.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
