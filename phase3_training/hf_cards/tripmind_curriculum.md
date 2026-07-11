---
language:
  - en
tags:
  - travel
  - india
  - curriculum-learning
  - llama
  - qlora
  - itinerary-optimization
  - grounding
license: apache-2.0
base_model: unsloth/Meta-Llama-3.1-8B
datasets:
  - ishreyadev/pivotai-synthetic-v2
  - ishreyadev/pivotai-agent-traces
metrics:
  - bertscore
model-index:
  - name: pivotai-curriculum
    results:
      - task:
          type: text-generation
          name: Travel Itinerary Optimization
        metrics:
          - type: grounding_accuracy
            value: 0.88
            name: Grounding Accuracy
          - type: bertscore_f1
            value: 0.734
            name: BERTScore F1
          - type: red_team_pass
            value: 0.60
            name: Red-Team Robustness
---

# pivotai-curriculum

Curriculum-trained Llama 3.1 8B for Indian domestic travel optimization. Uses **two-stage sequential training**: first on 4,749 Phase 1 synthetic pairs (domain knowledge), then on 449 Phase 2 agent reasoning traces (complex reasoning patterns).

Part of the [pivotai](https://github.com/ishreyadev/pivotai) project. The curriculum hypothesis was that domain knowledge should precede complex reasoning patterns — similar to how students learn fundamentals before advanced topics. Results revealed an interesting trade-off: the model achieved the **highest grounding accuracy (88%)** and **best red-team robustness (60%)** of the three variants, but the Phase 2 training stage catastrophically disrupted structured JSON output (10.9% validity).

## Model Details

| Property | Value |
|----------|-------|
| Base model | `unsloth/Meta-Llama-3.1-8B` |
| Training method | QLoRA r=8, α=16, dropout=0.05 (2-stage) |
| Stage 1 data | 4,749 pairs (Phase 1 synthetic) — 424 steps |
| Stage 2 data | 449 pairs (Phase 2 agent traces) — 171 steps |
| Final train loss | 0.313 (Stage 2) |
| Hardware | Lightning.ai A100 (bf16, seq_len=16384) |
| Format | GGUF Q4_K_M (4.6 GB) |

## Evaluation Results (92 test cases)

| Metric | Score | Target | ✓/✗ |
|--------|:-----:|:------:|:---:|
| JSON valid | **10.9%** | 85% | ✗ |
| Savings found | — | 70% | — |
| Schema compliance | 0.0% | 80% | ✗ |
| BERTScore F1 | 0.734 | 0.70 | ✓ |
| Intent alignment | 0.418 | 0.55 | ✗ |
| Grounding accuracy | **0.880** | 0.60 | ✓ |
| Reasoning coherence | 0.470 | 0.65 | ✗ |
| Red-team pass | **60.0%** | 80% | ✗ |

**Notable:** Despite near-zero JSON validity, grounding accuracy (0.88) nearly matches pivotai-ft (0.895). The model has absorbed real-world knowledge about Indian cities and travel patterns — it simply cannot format the output as valid JSON after Phase 2 training overwrote structured-output behavior.

**Recommendation:** Use with JSON-constrained decoding (llama.cpp `--grammar`, Outlines, or similar) to recover structured output. The underlying knowledge is strong.

## Usage with Ollama

```bash
ollama create pivotai-curriculum -f Modelfile.curriculum
ollama run pivotai-curriculum
```

**Note:** Due to low JSON validity in standard inference, consider using grammar-constrained decoding for reliable structured output.

## Limitations

- JSON validity is 10.9% — standard inference rarely produces valid JSON. Use grammar-constrained decoding.
- The Phase 2 curriculum stage appears to have overwritten Phase 1 structured-output training — a known curriculum learning failure mode.
- Despite strong semantic knowledge, the model cannot be used without output post-processing.
