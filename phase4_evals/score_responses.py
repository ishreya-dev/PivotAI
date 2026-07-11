"""
Phase 4 — Step 2 of 2: score saved model responses.
Reads responses_*.jsonl (from generate_responses.py) and computes all 9 metrics.
No Ollama calls — safe to rerun anytime to add metrics or fix bugs.

Usage:
    python phase4_evals/score_responses.py
    python phase4_evals/score_responses.py --responses-file data/evals/responses_20260611.jsonl
    python phase4_evals/score_responses.py --no-judge   # skip DeepSeek, local metrics only
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
from config import EVALS_DIR
from utils.logger import get_logger
from phase4_evals.schemas import EvalMetrics, EvalRecord
from phase4_evals.judge_prompts import REASONING_COHERENCE_PROMPT, GROUNDING_ACCURACY_PROMPT
from phase4_evals.metrics import (
    check_json_valid, check_savings_valid, check_budget_compliance,
    check_schema_compliance, score_intent_alignment,
    score_rouge_l, score_bertscore,
    call_llm_judge, parse_judge_score,
)

log = get_logger("phase4", "score_responses")


def _latest_responses_file() -> Path | None:
    candidates = sorted(EVALS_DIR.glob("responses_*.jsonl"))
    return candidates[-1] if candidates else None


def _score(response: dict, use_judge: bool) -> EvalRecord:
    model_name = response["model_name"]
    golden_record_id = response["golden_record_id"]
    persona = response.get("persona", {})
    raw_output = response.get("raw_output", "")
    ref_pair = response.get("ref_pair", {})
    reference_text = json.dumps(ref_pair, ensure_ascii=False) if ref_pair else ""

    json_valid, parsed = check_json_valid(raw_output)
    savings_valid = check_savings_valid(parsed, persona) if parsed else None
    budget_compliance = check_budget_compliance(parsed, persona) if parsed else None
    schema_compliance = check_schema_compliance(parsed) if parsed else 0.0
    intent_alignment = score_intent_alignment(persona, parsed) if parsed else None
    rouge_l = score_rouge_l(raw_output, reference_text) if reference_text else None
    bertscore_f1 = score_bertscore(raw_output, reference_text) if reference_text else None

    reasoning_coherence: float | None = None
    grounding_accuracy: float | None = None
    judge_raw = ""

    if use_judge and parsed and not raw_output.startswith("ERROR:"):
        destination = persona.get("destination_city", "")
        try:
            coh_raw = call_llm_judge(REASONING_COHERENCE_PROMPT.format(
                persona=json.dumps(persona, ensure_ascii=False),
                model_output=raw_output[:3000],
            ))
            reasoning_coherence = parse_judge_score(coh_raw)
            judge_raw += f"COHERENCE: {coh_raw}\n"
        except Exception as exc:
            log.warning("Coherence judge failed", error=str(exc))

        try:
            gnd_raw = call_llm_judge(GROUNDING_ACCURACY_PROMPT.format(
                destination_city=destination,
                model_output=raw_output[:3000],
            ))
            grounding_accuracy = parse_judge_score(gnd_raw)
            judge_raw += f"GROUNDING: {gnd_raw}"
        except Exception as exc:
            log.warning("Grounding judge failed", error=str(exc))

    metrics = EvalMetrics(
        json_valid=json_valid,
        savings_valid=savings_valid,
        budget_compliance=budget_compliance,
        schema_compliance=schema_compliance,
        intent_alignment=intent_alignment,
        rouge_l=rouge_l,
        bertscore_f1=bertscore_f1,
        reasoning_coherence=reasoning_coherence,
        grounding_accuracy=grounding_accuracy,
        judge_raw=judge_raw.strip(),
    )

    return EvalRecord(
        model_name=model_name,
        golden_record_id=golden_record_id,
        persona=persona,
        prompt_used=response.get("prompt_used", ""),
        raw_output=raw_output,
        parsed_output=parsed,
        metrics=metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score pivotai model responses")
    parser.add_argument("--responses-file", default=None, help="Specific responses file")
    parser.add_argument("--no-judge", action="store_true", help="Skip DeepSeek judge")
    args = parser.parse_args()

    resp_path = Path(args.responses_file) if args.responses_file else _latest_responses_file()
    if not resp_path or not resp_path.exists():
        print("ERROR: No responses file found. Run: python phase4_evals/generate_responses.py")
        sys.exit(1)

    # Load responses and attach golden record reference output
    golden_by_id: dict[str, dict] = {}
    golden_path = EVALS_DIR / "golden_set.jsonl"
    if golden_path.exists():
        for line in golden_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = rec.get("id") or rec.get("record_id", "")
                if rid:
                    golden_by_id[rid] = rec
            except Exception:
                continue

    responses = []
    for line in resp_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            # Skip Ollama errors
            if r.get("raw_output", "").startswith("ERROR:"):
                continue
            # Attach reference output from golden set
            ref = golden_by_id.get(r.get("golden_record_id", ""), {})
            r["ref_pair"] = ref.get("pair", {})
            responses.append(r)
        except Exception:
            continue

    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = EVALS_DIR / f"eval_results_{timestamp}.jsonl"

    print(f"Responses      : {len(responses)}")
    print(f"Judge          : {'disabled' if args.no_judge else 'DeepSeek deepseek-chat'}")
    print(f"Output         : {output_path}\n")

    json_valid_count = 0
    with tqdm(total=len(responses), desc="Scoring", unit="record") as pbar:
        for response in responses:
            record = _score(response, use_judge=not args.no_judge)
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
            if record.metrics.json_valid:
                json_valid_count += 1
            pbar.set_description(f"Scoring — json_valid={json_valid_count}/{pbar.n+1}")
            pbar.update(1)

    print(f"\n✓ Scored {len(responses)} responses — JSON valid: {json_valid_count}/{len(responses)}")
    print(f"  Output: {output_path}")
    print(f"\nNext: python phase4_evals/compare.py")
    log.info("Scoring complete", total=len(responses), json_valid=json_valid_count, output=str(output_path))


if __name__ == "__main__":
    main()
