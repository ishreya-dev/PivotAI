---
language:
  - en
tags:
  - travel
  - india
  - distillation
  - llama
  - qlora
  - itinerary-optimization
  - chain-of-thought
license: apache-2.0
base_model: unsloth/Meta-Llama-3.1-8B
datasets:
  - ishreyadev/pivotai-agent-traces
metrics:
  - bertscore
  - rouge
model-index:
  - name: pivotai-distill
    results:
      - task:
          type: text-generation
          name: Travel Itinerary Optimization
        metrics:
          - type: json_valid
            value: 0.924
            name: JSON Validity Rate
          - type: savings_valid
            value: 0.981
            name: Savings Found Rate
          - type: bertscore_f1
            value: 0.738
            name: BERTScore F1
          - type: reasoning_coherence
            value: 0.674
            name: Reasoning Coherence
---

# pivotai-distill

Knowledge-distilled Llama 3.1 8B for Indian domestic travel optimization. Distilled from 500 **multi-agent DeepSeek reasoning traces** (Phase 2 of the pivotai pipeline), where a Supervisor + Analyst + Concierge + Optimizer chain used real MCP tool calls to build itineraries.

Part of the [pivotai](https://github.com/ishreyadev/pivotai) project. Unlike `pivotai-ft` (trained on clean synthetic pairs), this model was trained on agent reasoning chains — the hypothesis being that richer teacher signal improves generalization. Results were mixed: reasoning coherence improved, but structural output compliance dropped.

## Model Details

| Property | Value |
|----------|-------|
| Base model | `unsloth/Meta-Llama-3.1-8B` |
| Training method | QLoRA r=8, α=16, dropout=0.05 |
| Training data | 449 Alpaca-format distillation pairs (Phase 2 agent traces) |
| Epochs | 5 |
| Final train loss | 0.429 |
| Hardware | Lightning.ai A100 (bf16, seq_len=16384) |
| Format | GGUF Q4_K_M (4.6 GB) |

The higher loss (0.429 vs 0.266 for ft) correlates with noisier training signal — agent traces include tool-call artifacts and variable output lengths that add training noise.

## Evaluation Results (92 test cases)

| Metric | Score | Target | ✓/✗ |
|--------|:-----:|:------:|:---:|
| JSON valid | 92.4% | 85% | ✓ |
| Savings found | 98.1% | 70% | ✓ |
| Budget compliance | — | 80% | — |
| Schema compliance | 0.0% | 80% | ✗ |
| BERTScore F1 | 0.738 | 0.70 | ✓ |
| ROUGE-L | 0.090 | 0.25 | ✗ |
| Reasoning coherence | 0.674 | 0.65 | ✓ |
| Grounding accuracy | 0.442 | 0.60 | ✗ |
| Red-team pass | 46.7% | 80% | ✗ |

Schema compliance of 0% indicates the model produces valid JSON but with a different structure than the expected schema — a consequence of the diverse output formats in the distillation training data.

## Usage with Ollama

```bash
ollama create pivotai-distill -f Modelfile.distill
ollama run pivotai-distill
```

Prompt format (Alpaca with reasoning chain instruction):
```
### Instruction:
Act as pivotai Supervisor for an Indian domestic trip. Coordinate the Analyst, Concierge, and Optimizer agents to find Price-Pivot Points and produce an optimized itinerary. Show the reasoning chain for each agent handoff, then provide the final pivot analysis and optimized itinerary.

### Input:
{"starting_city": "Mumbai", ...}

### Response:
```

## Limitations

- Schema compliance is 0% — produces valid JSON but in a non-standard structure.
- Not recommended for production use without post-processing to extract the itinerary.
- Trained on only 449 examples (vs 4,749 for ft) — limited coverage of edge cases.
