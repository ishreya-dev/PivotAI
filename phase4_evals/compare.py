"""
Phase 4 — Compare eval results across all 3 models.
Reads latest eval_results_*.jsonl + red_team_results_*.jsonl from data/evals/.
Prints a terminal comparison table and writes data/evals/summary_<timestamp>.json.

Usage:
    python phase4_evals/compare.py
    python phase4_evals/compare.py --eval-file data/evals/eval_results_20260610.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from config import EVALS_DIR, SLM_FT_MODEL, SLM_DIST_MODEL, SLM_CURRICULUM_MODEL, SLM_BASELINE_MODEL
from utils.logger import get_logger

log = get_logger("phase4", "compare")

ALL_MODELS = [SLM_FT_MODEL, SLM_DIST_MODEL, SLM_CURRICULUM_MODEL, SLM_BASELINE_MODEL]

# Target thresholds
_TARGETS = {
    "json_valid":           0.85,
    "savings_valid":        0.70,
    "budget_compliance":    0.80,
    "schema_compliance":    0.80,
    "intent_alignment":     0.55,
    "rouge_l":              0.25,
    "bertscore_f1":         0.70,
    "reasoning_coherence":  0.65,
    "grounding_accuracy":   0.60,
    "red_team_pass":        0.80,
}

_METRIC_LABELS = {
    "json_valid":           "JSON valid",
    "savings_valid":        "Savings found",
    "budget_compliance":    "Budget compliance",
    "schema_compliance":    "Schema compliance",
    "intent_alignment":     "Intent alignment",
    "rouge_l":              "ROUGE-L (vs teacher)",
    "bertscore_f1":         "BERTScore F1 (vs teacher)",
    "reasoning_coherence":  "Reasoning coherence",
    "grounding_accuracy":   "Grounding accuracy",
    "red_team_pass":        "Red team pass",
}

_N_A = "n/a"


# ── File discovery ─────────────────────────────────────────────────────────────

def _latest_file(pattern: str) -> Path | None:
    candidates = sorted(EVALS_DIR.glob(pattern))
    return candidates[-1] if candidates else None


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# ── Aggregation ───────────────────────────────────────────────────────────────

def _aggregate_evals(records: list[dict]) -> dict[str, dict]:
    """Returns {model_name: {metric: value}}."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        buckets[r.get("model_name", "unknown")].append(r)

    result: dict[str, dict] = {}
    for model, recs in buckets.items():
        metrics_list = [r.get("metrics", {}) for r in recs]
        n = len(metrics_list)

        def _rate(key: str) -> float | None:
            vals = [m[key] for m in metrics_list if m.get(key) is not None]
            return sum(1 for v in vals if v) / len(vals) if vals else None

        def _avg(key: str) -> float | None:
            vals = [m[key] for m in metrics_list if isinstance(m.get(key), (int, float))]
            return round(sum(vals) / len(vals), 4) if vals else None

        result[model] = {
            "n": n,
            "json_valid": _rate("json_valid"),
            "savings_valid": _rate("savings_valid"),
            "budget_compliance": _rate("budget_compliance"),
            "schema_compliance": _avg("schema_compliance"),
            "intent_alignment": _avg("intent_alignment"),
            "rouge_l": _avg("rouge_l"),
            "bertscore_f1": _avg("bertscore_f1"),
            "reasoning_coherence": _avg("reasoning_coherence"),
            "grounding_accuracy": _avg("grounding_accuracy"),
        }

    return result


def _aggregate_redteam(records: list[dict]) -> dict[str, dict]:
    """Returns {model_name: {red_team_pass: float, n: int}}."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        buckets[r.get("model_name", "unknown")].append(r)

    result: dict[str, dict] = {}
    for model, recs in buckets.items():
        passed = [r for r in recs if r.get("passed", False)]
        result[model] = {
            "n": len(recs),
            "red_team_pass": round(len(passed) / len(recs), 4) if recs else None,
        }
    return result


# ── Formatting ────────────────────────────────────────────────────────────────

_AVG_METRICS = {
    "intent_alignment", "reasoning_coherence", "grounding_accuracy",
    "schema_compliance", "rouge_l", "bertscore_f1",
}


def _fmt(value: float | None, metric: str, target: float) -> str:
    if value is None:
        return _N_A
    formatted = f"{value:.3f}" if metric in _AVG_METRICS else f"{value*100:.1f}%"
    flag = " ✓" if value >= target else " ✗"
    return formatted + flag


# ── Print table ───────────────────────────────────────────────────────────────

def _print_table(eval_data: dict, rt_data: dict, models: list[str]) -> None:
    col_w = 22
    label_w = 24

    header = f"{'Metric':<{label_w}}"
    for m in models:
        short = m.replace("pivotai-", "")
        header += f"  {short:>{col_w}}"
    header += f"  {'Target':>{col_w}}"
    sep = "-" * (label_w + (col_w + 2) * (len(models) + 1))

    print("\n" + sep)
    print(header)
    print(sep)

    for metric_key, label in _METRIC_LABELS.items():
        target = _TARGETS[metric_key]
        row = f"{label:<{label_w}}"
        for m in models:
            if metric_key == "red_team_pass":
                val = rt_data.get(m, {}).get("red_team_pass")
            else:
                val = eval_data.get(m, {}).get(metric_key)
            row += f"  {_fmt(val, metric_key, target):>{col_w}}"
        row += f"  {_fmt(target, metric_key, target):>{col_w}}"
        print(row)

    print(sep)

    # Sample counts
    row = f"{'Records evaluated':<{label_w}}"
    for m in models:
        n = eval_data.get(m, {}).get("n", 0)
        row += f"  {str(n):>{col_w}}"
    row += f"  {'100':>{col_w}}"
    print(row)

    row = f"{'Red team runs':<{label_w}}"
    for m in models:
        n = rt_data.get(m, {}).get("n", 0)
        row += f"  {str(n):>{col_w}}"
    row += f"  {'50':>{col_w}}"
    print(row)

    print(sep + "\n")


# ── Head-to-head ──────────────────────────────────────────────────────────────

def _run_head_to_head(eval_records: list[dict], models: list[str]) -> dict:
    """For each golden record, compare every pair of models via LLM judge.
    Returns {(model_a, model_b): {"wins_a": int, "wins_b": int, "ties": int}}.
    Cached via call_llm_judge's @api_cache so reruns are instant.
    """
    from itertools import combinations
    from phase4_evals.metrics import call_llm_judge
    from phase4_evals.judge_prompts import HEAD_TO_HEAD_PROMPT

    # Group outputs by golden_record_id
    by_record: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in eval_records:
        by_record[r["golden_record_id"]][r["model_name"]] = r

    results: dict[tuple, dict] = {
        (a, b): {"wins_a": 0, "wins_b": 0, "ties": 0}
        for a, b in combinations(models, 2)
    }

    total_pairs = len(by_record) * len(list(combinations(models, 2)))
    done = 0
    print(f"\nHead-to-head: {total_pairs} judge calls...")

    for record_id, model_outputs in by_record.items():
        for model_a, model_b in combinations(models, 2):
            out_a = model_outputs.get(model_a, {}).get("raw_output", "")
            out_b = model_outputs.get(model_b, {}).get("raw_output", "")
            persona = next(iter(model_outputs.values()), {}).get("persona", {})
            if not out_a or not out_b:
                continue
            try:
                prompt = HEAD_TO_HEAD_PROMPT.format(
                    persona=json.dumps(persona, ensure_ascii=False),
                    model_a=model_a, output_a=out_a[:1500],
                    model_b=model_b, output_b=out_b[:1500],
                )
                raw = call_llm_judge(prompt)
                winner = json.loads(raw).get("winner", "tie")
            except Exception:
                winner = "tie"
            pair = (model_a, model_b)
            if winner == "model_a":
                results[pair]["wins_a"] += 1
            elif winner == "model_b":
                results[pair]["wins_b"] += 1
            else:
                results[pair]["ties"] += 1
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{total_pairs} comparisons done")

    return results


def _print_head_to_head(h2h: dict, models: list[str]) -> None:
    from itertools import combinations
    print("\nHEAD-TO-HEAD WIN RATES")
    print("-" * 52)
    for model_a, model_b in combinations(models, 2):
        res = h2h.get((model_a, model_b), {})
        wa = res.get("wins_a", 0)
        wb = res.get("wins_b", 0)
        t = res.get("ties", 0)
        total = wa + wb + t or 1
        short_a = model_a.replace("pivotai-", "")
        short_b = model_b.replace("pivotai-", "")
        print(f"  {short_a} vs {short_b}: "
              f"{short_a} wins {wa/total*100:.0f}% | "
              f"{short_b} wins {wb/total*100:.0f}% | "
              f"tie {t/total*100:.0f}%  (n={total})")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="pivotai eval comparison")
    parser.add_argument("--eval-file", default=None, help="Specific eval results file")
    parser.add_argument("--rt-file", default=None, help="Specific red team results file")
    parser.add_argument("--no-h2h", action="store_true", help="Skip head-to-head judge calls")
    args = parser.parse_args()

    # Find files
    eval_path = Path(args.eval_file) if args.eval_file else _latest_file("eval_results_*.jsonl")
    rt_path = Path(args.rt_file) if args.rt_file else _latest_file("red_team_results_*.jsonl")

    if not eval_path or not eval_path.exists():
        print("ERROR: No eval results found. Run: python phase4_evals/run_evals.py")
        sys.exit(1)
    if not rt_path or not rt_path.exists():
        print("WARNING: No red team results found. Run: python phase4_evals/red_team.py")
        rt_records: list[dict] = []
    else:
        rt_records = _load_jsonl(rt_path)

    eval_records = _load_jsonl(eval_path)

    eval_data = _aggregate_evals(eval_records)
    rt_data = _aggregate_redteam(rt_records)

    # Ensure all models are present (even if no records)
    for m in ALL_MODELS:
        eval_data.setdefault(m, {})
        rt_data.setdefault(m, {})

    print(f"\nEval file:      {eval_path.name}")
    print(f"Red team file:  {rt_path.name if rt_path else 'none'}")

    _print_table(eval_data, rt_data, ALL_MODELS)

    # Head-to-head win rates
    h2h: dict = {}
    if not args.no_h2h and eval_records:
        h2h = _run_head_to_head(eval_records, ALL_MODELS)
        _print_head_to_head(h2h, ALL_MODELS)
    else:
        print("(head-to-head skipped — run without --no-h2h to include)")

    # Write summary JSON for Phase 5 system health page
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_path = EVALS_DIR / f"summary_{timestamp}.json"
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_file": str(eval_path),
        "red_team_file": str(rt_path) if rt_path else None,
        "targets": _TARGETS,
        "models": {},
    }
    from itertools import combinations
    h2h_summary = {}
    for model_a, model_b in combinations(ALL_MODELS, 2):
        res = h2h.get((model_a, model_b), {})
        total = sum(res.values()) or 1
        h2h_summary[f"{model_a}_vs_{model_b}"] = {
            "wins_a": res.get("wins_a", 0),
            "wins_b": res.get("wins_b", 0),
            "ties": res.get("ties", 0),
            "win_rate_a": round(res.get("wins_a", 0) / total, 4),
        }

    for m in ALL_MODELS:
        summary["models"][m] = {
            **eval_data.get(m, {}),
            "red_team_pass": rt_data.get(m, {}).get("red_team_pass"),
            "red_team_n": rt_data.get(m, {}).get("n", 0),
        }
    summary["head_to_head"] = h2h_summary

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary written: {summary_path}")
    log.info("Comparison complete", summary=str(summary_path))


if __name__ == "__main__":
    main()
