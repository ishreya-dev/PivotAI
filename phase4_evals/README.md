# Phase 4 — Evaluation & Red Teaming

Benchmarks all four models (3 fine-tuned + 1 untuned baseline) on 92 golden test cases and 45 adversarial red-team prompts. Measures structural correctness, semantic quality, real-world grounding, and robustness to adversarial inputs.

---

## Models Evaluated

| Model | Training | Records |
|-------|----------|---------|
| `llama3.1:8b` | Untuned baseline — pre-training floor | 92 |
| `pivotai-ft` | SFT on 4,749 Phase 1 synthetic pairs | 92 |
| `pivotai-distill` | Distilled from 449 Phase 2 agent traces | 92 |
| `pivotai-curriculum` | Two-stage: Phase 1 → Phase 2 sequential | 92 |

---

## Evaluation Metrics

| Metric | How it is computed |
|--------|--------------------|
| **JSON valid** | `json.loads()` — does the model produce parseable JSON? |
| **Savings found** | `optimized_cost < baseline_cost` — does the output find a cheaper option? |
| **Budget compliance** | Per-person daily cost within ±20% of the persona's budget tier bounds |
| **Schema compliance** | Fraction of required keys present: `optimized`, `pivot_analysis`, `total_trip_cost`, `daily_itinerary` |
| **ROUGE-L** | N-gram overlap with reference output — rewards format fidelity |
| **BERTScore F1** | Semantic embedding similarity (DistilBERT) — rewards content accuracy |
| **Intent alignment** | Cosine similarity between stated intents and itinerary activities via `all-MiniLM-L6-v2` |
| **Reasoning coherence** | LLM judge (DeepSeek V4 Flash) — is the pivot explanation logical? |
| **Grounding accuracy** | LLM judge (DeepSeek V4 Flash) — do the itinerary details reflect real Indian cities/costs? |
| **Red-team pass** | Does the model refuse to violate budget constraints under adversarial prompts? |

---

## Results Summary

| Metric | baseline | pivotai-ft | pivotai-distill | pivotai-curriculum |
|--------|:--------:|:-----------:|:----------------:|:-------------------:|
| JSON valid | 0.0% | **100%** | 92.4% | 10.9% |
| Savings found | — | **100%** | 98.1% | — |
| Budget compliance | — | **98.7%** | — | — |
| Schema compliance | 0.0% | **83.7%** | 0.0% | 0.0% |
| ROUGE-L | 12.6% | **43.6%** | 8.9% | 12.7% |
| BERTScore F1 | 80.5%† | **93.2%** | 73.8% | 73.4% |
| Intent alignment | — | 32.2% | — | 41.8% |
| Reasoning coherence | — | **72.3%** | 67.4% | 47.0% |
| Grounding accuracy | —‡ | **89.5%** | 44.2% | **88.0%** |
| Red-team pass | —§ | 53.3% | 46.7% | **60.0%** |

† BERTScore misleadingly scores the baseline high despite 0% JSON validity — semantic similarity is not a reliable signal for structured-output tasks. ROUGE-L (12.6%) correctly captures the gap.  
‡ Grounding accuracy requires the LLM judge on parsed output — not computed for baseline due to 0% JSON validity.  
§ Red-team adversarial evaluation not run on baseline — with 0% JSON validity, structured safety metrics are not interpretable.

Full narrative and analysis: [RESULTS.md](../RESULTS.md)

---

## Files

| File | Purpose |
|------|---------|
| `build_golden_set.py` | Sample 92 diverse test cases from Phase 1 data |
| `generate_responses.py` | Run all 4 models against the golden set via Ollama |
| `score_responses.py` | Compute structural + semantic metrics locally |
| `metrics.py` | Metric implementations (ROUGE-L, BERTScore, intent alignment, schema) |
| `judge_prompts.py` | LLM judge prompts for reasoning coherence + grounding accuracy |
| `compare.py` | Aggregate scores, compute head-to-head win rates, write summary JSON |
| `red_team.py` | Run 45 adversarial prompts and score refusal behaviour |
| `utils.py` | Shared helpers: `build_prompt`, `call_ollama`, `NUM_PREDICT` per model |
| `schemas.py` | Pydantic models for eval records |
| `run_evals.py` | End-to-end runner (calls generate → score → compare in order) |

---

## Running the Eval Pipeline

Requires Ollama running with all 4 models loaded.

```bash
# Register models (run once from project root)
ollama create pivotai-ft         -f phase3_training/notebooks/modelfiles/Modelfile.ft
ollama create pivotai-distill    -f phase3_training/notebooks/modelfiles/Modelfile.distill
ollama create pivotai-curriculum -f phase3_training/notebooks/modelfiles/Modelfile.curriculum
# llama3.1:8b is pulled directly: ollama pull llama3.1:8b

# Full pipeline
python phase4_evals/run_evals.py

# Or step by step
python phase4_evals/generate_responses.py   # ~3 hours on MacBook Air CPU
python phase4_evals/score_responses.py
python phase4_evals/compare.py
python phase4_evals/red_team.py
```

For the baseline model only, use the Colab notebook `notebooks/05_baseline_comparison.ipynb` — inference for 92 cases at ~60s each takes ~3 hours on CPU.

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `04_generate_responses.ipynb` | Colab notebook for running eval inference (all 4 models) |
| `05_baseline_comparison.ipynb` | Self-contained Colab notebook for baseline-only run |
| `results_analysis.ipynb` | Generates 4 comparison charts saved to `data/evals/charts/` |

---

## Output Files

All outputs in `data/evals/`:

| File | Contents |
|------|---------|
| `golden_set.jsonl` | 92 golden test cases (committed to repo) |
| `responses_*.jsonl` | Raw model outputs per run |
| `eval_results_*.jsonl` | Per-record scored metrics |
| `red_team_results_*.jsonl` | Red-team adversarial results |
| `baseline_responses.jsonl` | Baseline model raw outputs |
| `baseline_scores.jsonl` | Baseline model scored metrics |
| `summary_*.json` | Aggregated metrics + head-to-head win rates |
| `charts/` | 4 comparison charts (PNG) |
