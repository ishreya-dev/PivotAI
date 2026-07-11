# ItinerAI-Bench — Evaluation Results

Benchmark report for three Llama 3.1 8B variants, each fine-tuned with a different supervision signal for Indian domestic travel itinerary optimization.

---

## Overview

Three Llama 3.1 8B models were trained on the same task using different training signals, then evaluated against each other and an untuned baseline to answer one question:

> **Does richer teacher signal from agent reasoning traces produce a better travel optimizer than plain supervised fine-tuning on synthetic pairs?**

The short answer: no — plain SFT on clean synthetic pairs won on structure, correctness, and head-to-head judging. Distillation and curriculum learning each surfaced their own failure modes, detailed below.

---

## Experimental Setup

### Models

| Model | Signal | Hypothesis under test |
|---|---|---|
| `llama3.1:8b` (baseline) | None (untuned) | Establishes the pre-training floor |
| `itinerai-ft` | SFT on 5,000 synthetic pairs (GPT-4o-mini teacher) | Clean, validated pairs are sufficient |
| `itinerai-distill` | Distilled from 500 DeepSeek agent reasoning traces | Multi-step tool-calling exposure improves generalization |
| `itinerai-curriculum` | Two-stage: Phase 1 pairs → Phase 2 traces | Sequential domain-then-reasoning training beats either alone |

### Dataset

Dataset size was set by **budget parity**, not convenience:

- **Phase 1** — GPT-4o-mini generated 5,000 validated `<persona, optimized_itinerary>` pairs for **$4**
- **Phase 2** — DeepSeek multi-agent pipeline generated 500 reasoning traces (4 agents, 3–5 real MCP tool calls each) for **$4**

Equal spend, different data strategies — the ft vs. distill comparison measures signal quality, not data scale. Total data cost: **$8**. Compute, inference, and all other APIs were free.

### Evaluation Methodology

- **92 golden test cases**, run against all 4 models
- **45 adversarial red-team prompts** (budget bypass, constraint violation, prompt injection) — not run on baseline, since 0% JSON validity makes structured safety metrics uninterpretable
- Judging combines automated structural checks, embedding-based semantic similarity, and LLM-as-judge scoring

---

## Benchmark Results

92 golden test cases × 4 models.

| Metric | baseline | itinerai-ft | itinerai-distill | itinerai-curriculum |
|---|:---:|:---:|:---:|:---:|
| JSON valid | 0.0% | **100%** | 92.4% | 10.9% |
| Savings found | — | **100%** | 98.1% | — |
| Budget compliance | — | **98.7%** | — | — |
| Schema compliance | 0.0% | **83.7%** | 0.0% | 0.0% |
| Intent alignment | — | 32.2% | — | 41.8% |
| ROUGE-L | 12.6% | **43.6%** | 8.9% | 12.7% |
| BERTScore F1 | 80.5%¹ | **93.2%** | 73.8% | 73.4% |
| Reasoning coherence | — | **72.3%** | 67.4% | 47.0% |
| Grounding accuracy | —² | **89.5%** | 44.2% | **88.0%** |
| Red-team pass | —³ | 53.3% | 46.7% | **60.0%** |

<sub>¹ High despite 0% JSON validity — see Notes. ² Requires the LLM judge on parsed output; not computed for baseline. ³ Not run on baseline; uninterpretable at 0% JSON validity.</sub>

> **Key Takeaways**
> - **Best structural model:** `itinerai-ft` — 100% JSON validity, 100% savings found, 98.7% budget compliance
> - **Best grounding:** `itinerai-ft` (89.5%), with `itinerai-curriculum` close behind (88.0%) despite near-zero JSON validity
> - **Best red-team robustness:** `itinerai-curriculum` (60.0%) — still well below the 80% target
> - **Biggest failure mode:** `itinerai-curriculum` collapsed to 10.9% JSON validity after Phase 2 training, despite starting from a checkpoint that produced 100% valid JSON
> - **Metric blind spot:** baseline BERTScore (80.5%) scores as if competent despite 0% JSON validity — ROUGE-L (12.6%) is the metric that correctly penalizes it

---

## Pairwise Comparison

Head-to-head LLM-judge scoring on the same 92 records:

| Matchup | Result |
|---|---|
| ft vs. distill | **ft wins 78%** (72/92) |
| ft vs. curriculum | **ft wins 57%** (52/92) |
| distill vs. curriculum | distill wins 52% (48/92) — statistical tie |

`itinerai-ft` is the clear winner against both alternative training strategies, decisively against distillation and more narrowly against curriculum learning. The ft vs. curriculum margin (57/43) is close enough that it should not be treated as conclusive at n=92 — see [Limitations](#limitations).

The distill vs. curriculum result is functionally a coin flip, indicating neither alternative strategy produced a consistently better optimizer than the other, even though they fail in different ways (distill: weaker grounding; curriculum: broken output structure).

---

## Analysis

**1. Fine-tuning on clean pairs wins on structural correctness**

- **Observation:** `itinerai-ft` leads on every structural metric — JSON validity, savings found, budget compliance, schema compliance.
- **Evidence:** 100% JSON valid / 100% savings found / 98.7% budget compliance, versus 92.4% / 98.1% / — for distillation.
- **Interpretation:** training directly on validated `<persona, itinerary>` pairs teaches the exact output contract; exposure to free-form agent reasoning during distillation appears to introduce format noise instead of improving generalization.
- **Engineering implication:** when the task has a strict output schema, prioritize clean supervised pairs over richer-but-noisier reasoning traces, unless the traces are post-processed to isolate the final structured answer.

**2. Curriculum learning's second stage overwrote first-stage structure**

- **Observation:** `itinerai-curriculum` collapsed to 10.9% JSON validity, despite initializing from Phase 1 weights that independently reached 100%.
- **Evidence:** Phase 2 traces are long multi-agent reasoning chains; the model appears to have learned to reproduce that format instead of compact JSON.
- **Interpretation:** this is a known curriculum-learning failure mode — later stages can unlearn skills established earlier if the output distributions differ sharply.
- **Engineering implication:** curriculum stages need either output-format consistency across stages or constrained decoding to protect the schema learned early on.

**3. Curriculum grounding held up despite broken formatting**

- **Observation:** grounding accuracy for curriculum (88.0%) is nearly identical to ft (89.5%), despite the JSON collapse.
- **Evidence:** the model still produces factually grounded content about cities, transit, and pricing — it just fails to package it correctly.
- **Interpretation:** the underlying domain knowledge transferred from Phase 2 traces; the failure is a formatting regression, not a knowledge regression.
- **Engineering implication:** grammar-constrained decoding (e.g., Outlines, llama.cpp grammars) is a plausible low-cost fix to recover most of this model's value without retraining.

**4. All three models fail red-team robustness**

- **Observation:** none of the three fine-tuned models reach the 80% red-team pass target (ft: 53.3%, distill: 46.7%, curriculum: 60.0%).
- **Evidence:** adversarial cases cover budget bypass, constraint violation, and prompt injection.
- **Interpretation:** SFT optimizes for in-distribution correctness, not for rejecting out-of-distribution or adversarial instructions — this gap is expected, not anomalous.
- **Engineering implication:** robustness requires a dedicated training signal (e.g., adversarial preference data) rather than emerging from task-focused SFT.

**5. Intent alignment is weak across the board**

- **Observation:** all models score well below the 0.55 intent alignment target (ft: 32.2%, curriculum: 41.8%).
- **Evidence:** measured via sentence-transformer cosine similarity between optimized itinerary and stated traveler intents (e.g., Adventure, Nightlife).
- **Interpretation:** models consistently prioritize cost minimization over activity-level personalization.
- **Engineering implication:** the Phase 1 training data likely over-weighted cost savings as the reward signal; future data generation should explicitly balance intent fidelity alongside savings.

---

## Notes

- **BERTScore blind spot:** baseline BERTScore (80.5%) is high despite 0% JSON validity, because embedding similarity rewards natural language that mentions the right entities even when totally unstructured. ROUGE-L (12.6%) is the more reliable differentiator for structured-output tasks, since n-gram overlap penalizes format mismatches that embeddings miss.
- **Grounding accuracy** and **red-team pass** are not computed for the baseline — both require parseable structured output, which the baseline does not produce.

---

## Limitations

- **Evaluation set size.** 92 cases is workable for headline comparisons but marginal for close results — the 57/43 ft-vs-curriculum margin is not statistically significant at this sample size.
- **Judge dependence.** Reasoning coherence, grounding accuracy, and pairwise comparisons rely on LLM-as-judge scoring (DeepSeek V4 Flash), which carries its own biases and is not independently verified against human annotation.
- **Red-team coverage.** 45 adversarial prompts test a limited slice of possible attack surface; passing rates here should be read as a lower bound, not a comprehensive robustness score.
- **Metric interpretability at 0% JSON validity.** Several metrics (savings found, budget compliance, grounding, red-team pass) are undefined for the baseline and for curriculum's near-collapsed output, which limits direct comparability on those rows.

---

## Future Improvements

**1. Fix curriculum JSON collapse**
- *Problem:* Phase 2 training overwrites the structured-output behavior learned in Phase 1.
- *Proposed solution:* apply grammar-constrained decoding (llama.cpp `--grammar` or Outlines) during Phase 2, or add a short JSON-only warmup pass after Phase 2 completes.
- *Expected impact:* recovers curriculum's strong grounding accuracy (88.0%) without sacrificing schema compliance.

**2. Improve red-team robustness**
- *Problem:* all three models fail adversarial prompts at well below the 80% target.
- *Proposed solution:* build a 500-case adversarial dataset and apply Direct Preference Optimization (DPO), using the safe response as chosen and the unsafe response as rejected.
- *Expected impact:* targets robustness directly rather than relying on it to emerge from task-focused SFT.

**3. Improve intent alignment**
- *Problem:* all models under-express activity-level personalization relative to cost savings.
- *Proposed solution:* augment Phase 1 prompts to explicitly weight intent fidelity alongside cost savings in the pivot analysis.
- *Expected impact:* should raise intent alignment scores across future fine-tunes without touching structural metrics.

**4. Expand the evaluation set**
- *Problem:* 92 cases gives insufficient confidence on close matchups (ft vs. curriculum).
- *Proposed solution:* grow the golden set to 500+ cases.
- *Expected impact:* tighter confidence intervals on head-to-head results, making close comparisons statistically meaningful.

---

## Training Efficiency

| Model | Final Train Loss | Steps | Hardware | Time |
|---|:---:|:---:|:---:|:---:|
| itinerai-ft | **0.266** | 636 | Colab T4 | ~1.8h |
| itinerai-distill | 0.429 | 285 | Lightning.ai A100 | ~2.8h |
| itinerai-curriculum | 0.313 | 424 + 171 | Lightning.ai A100 | ~3.5h |

`itinerai-ft` reached the lowest loss on the fewest steps, consistent with its eval-time lead — clean, consistent training pairs are highly sample-efficient. Distillation's higher loss (0.429) tracks with its noisier output schema at eval time: more training examples (4,749 vs. 449) outweighed the richer per-example signal that distillation was intended to provide.

---

## Final Summary

**Did fine-tuning work?** Yes, decisively. `itinerai-ft` moved JSON validity from 0% to 100% and led on every structural and grounding metric.

**Did knowledge distillation help?** No, not at this scale. 449 distilled examples underperformed 4,749 plain SFT pairs on nearly every metric, and lost 78% of head-to-head judged comparisons against ft.

**Did curriculum learning help?** Partially. It produced the best red-team pass rate and grounding nearly matching ft, but its second training stage broke JSON structure almost entirely — a net negative until the JSON-collapse issue is fixed.

**Biggest engineering lesson:** when an LLM task has a strict output contract, protecting that contract across every training stage matters more than the richness of the training signal. Reasoning quality and grounding can transfer through distillation or curriculum training even when structural compliance collapses — meaning schema enforcement should be treated as its own training objective, not an assumed side effect of good data.