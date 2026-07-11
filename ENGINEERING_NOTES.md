# Engineering Decisions

This document explains the constraints, trade-offs, and design decisions that shaped ItinerAI-Bench — not a chronological log of what broke, but a record of why the system is built the way it is.

---

## Engineering Principles

- **Free-tier first.** Every external dependency was chosen to run within a free tier at batch scale, not just for interactive use.
- **Reproducibility over convenience.** Anything expensive or non-deterministic to regenerate (synthetic data, agent traces) is documented and reproducible from source, rather than committed as a static artifact.
- **Standards-based integration.** Tools are exposed through MCP rather than bespoke wrappers, so they work unmodified in other MCP-compatible clients.
- **Local inference where possible.** Final models run as quantized GGUFs via Ollama — no GPU dependency at serving time.
- **Heuristics are acceptable when documented.** Where no free ground-truth data source existed, a transparent estimator was used instead of blocking the project or silently degrading data quality.
- **Findings over silent fixes.** Some failures (curriculum JSON collapse, BERTScore blind spot) were left as documented findings rather than patched, because the fix was out of scope and the failure itself was informative.
- **Data generation is checkpointed and idempotent.** Long-running, paid API jobs are designed to resume, not restart.

---

## External Services

Several components depend on third-party data sources. The overarching constraint: batch agent workloads (hundreds of traces, thousands of tool calls) exceed what most free-tier APIs are designed for.

### Challenge
No free API provided live flight pricing for Indian domestic routes. Available options required paid accounts or partner-level access, and none offered rate limits usable for 500+ concurrent agent traces.

### Design Decision
Transit cost is estimated with a haversine distance calculation — straight-line distance between city coordinates multiplied by a per-km rate from `config.BUDGET_TIERS` — rather than sourced from a live pricing API.

### Trade-off
This is a heuristic, not real-world pricing. It sacrifices pricing realism for reproducibility and zero marginal cost, and is documented as a known limitation on the model cards.

### Outcome
The optimizer's cost analysis remains internally consistent and fully reproducible without any paid API dependency.

---

### Challenge
Free-tier rate limits on Overpass, OpenRouteService, and DuckDuckGo are designed for interactive use. The agent pipeline required an estimated 6,000–10,000 external calls across 500 traces (4 agents × 3–5 calls each).

### Design Decision
Every MCP server applies a 24-hour response cache (`@api_cache`, backed by `diskcache`). DuckDuckGo was selected for web search specifically because it requires no API key and imposes no hard rate limit, in preference to paid alternatives like Tavily or Brave Search.

### Trade-off
Cached responses can go slightly stale within the 24-hour window, which is acceptable for route/hotel/POI data that doesn't change meaningfully at that timescale.

### Outcome
The 20-city travel network has only ~380 unique city pairs, so caching collapsed thousands of agent calls into roughly 40 real upstream requests — comfortably within every free tier in use.

---

## Data Generation

### Challenge
The GPT-4o-mini teacher model occasionally produced itineraries priced outside the traveler's stated budget tier, which would silently corrupt downstream training data if left unchecked.

### Design Decision
A three-gate validator (`phase1_data_engine/validate.py`) checks JSON structure validity, cost within ±20% of budget-tier bounds, and a minimum 5% savings in the proposed pivot. Records failing any gate are discarded and regenerated.

### Trade-off
The ~12% rejection rate increases API spend per accepted record, but this cost was accepted to guarantee internal consistency across all 5,000 training pairs.

### Outcome
All 5,000 Phase 1 pairs meet the same validity bar, giving the downstream fine-tune a clean, consistent training signal.

---

### Challenge
Generating 5,000 records at roughly 1.5 requests/second takes hours, and a mid-run crash would waste already-spent API budget.

### Design Decision
Each validated record is appended to disk immediately, and the generator loads existing output on startup to skip already-completed IDs.

### Trade-off
None of consequence — the append-on-write pattern adds negligible overhead.

### Outcome
The generation job is fully resumable; a crash costs at most the in-flight record, not the run.

---

## Agent System

The multi-agent pipeline (Supervisor → Analyst → Concierge → Optimizer) is the core of Phase 2. Its design decisions center on tool standardization and trace quality.

### Challenge
DeepSeek's function-calling response format differs subtly from OpenAI's — occasionally omitting the `tool_call.id` field or nesting arguments differently — despite the agent chain being built against the OpenAI tool-call spec.

### Design Decision
A normalization layer in `mcp_adapter.py` reconciles these differences before tool results are returned to the agent chain, allowing the same agent code to work against either provider.

### Trade-off
Adds a small amount of adapter complexity in exchange for provider portability.

### Outcome
The agent chain is decoupled from any single LLM provider's function-calling quirks.

---

### Challenge
Roughly 8% of raw agent traces exhibited degenerate behavior — looping tool calls or an empty final optimizer response — which would have poisoned the distillation dataset if included.

### Design Decision
A quality filter discards traces with more than 6 tool calls per agent, duplicate consecutive tool calls, or an empty JSON body in the final optimizer response.

### Trade-off
This reduced the usable trace count from 545 to 500, a real cost against the fixed $4 generation budget.

### Outcome
The 500 traces used for distillation and curriculum training are free of the degenerate patterns observed in the raw output.

---

> **Why MCP over direct API wrappers:** Exposing routing, hotels, POIs, and search as MCP servers — rather than ad hoc SDK calls — means the same four tools work unmodified inside Claude Desktop, Claude Code, or any other MCP-compatible agent. The protocol boundary also kept agent prompts and tool schemas clean and independently testable.

---

## Model Training

Training spanned three supervision strategies (SFT, distillation, curriculum) across two hardware environments, chosen to match each dataset's sequence-length requirements.

### Challenge
Distillation and curriculum training required a 16,384-token sequence length to fit full agent reasoning traces — well beyond what fits comfortably in a free-tier Colab T4 session (15GB VRAM, 12-hour limit, no OOM/idle-reclaim guarantees).

### Design Decision
Distillation and curriculum training moved to Lightning.ai A100 (40GB VRAM, stable 3-hour free session). The shorter-sequence `itinerai-ft` run (seq_len=512) stayed on Colab T4, where fp16 precision and gradient accumulation of 4 kept peak memory to ~11GB.

### Trade-off
Splitting training across two platforms adds environment-management overhead, but matching hardware to sequence length was necessary to complete any of the longer runs at all.

### Outcome
All three models trained to completion within free-tier compute, with per-epoch checkpointing limiting any interruption to at most one epoch of lost progress.

---

### Challenge
The curriculum model's second training stage (Phase 2 traces) reduced JSON validity from 100% — achieved after Stage 1 — to 10.9%, despite initializing from a checkpoint that already satisfied the output contract.

### Design Decision
This was documented as a finding rather than patched. A proper fix — grammar-constrained decoding during Stage 2, or a brief JSON-only warmup pass afterward — was identified but fell outside the project's compute budget. The curriculum model is recommended for use only with grammar-constrained decoding.

### Trade-off
Leaving the model as-is means `itinerai-curriculum` is not usable in production without an additional decoding constraint layer.

### Outcome
The result stands as a clear, reproducible demonstration of a known curriculum-learning failure mode — a second training stage can overwrite structural behavior learned in the first when the two stages' output distributions diverge sharply.

---

### Challenge
Standard HuggingFace export tooling does not produce Ollama-compatible GGUF files directly.

### Design Decision
llama.cpp was built from source for GGUF conversion, quantized at `Q4_K_M` to balance model quality against the 4.6GB per-model size budget. Unsloth's `model.save_pretrained_gguf()` shortcut was adopted on the A100 notebooks as the preferred path.

### Trade-off
Q4_K_M sacrifices some numerical precision relative to higher-bit quantizations, in exchange for a size small enough to run comfortably on consumer hardware.

### Outcome
All three models run locally via Ollama with no GPU required at inference time.

---

## Evaluation

### Challenge
Running the LLM-as-judge pass (DeepSeek V4 Flash) across 92 cases × 3 models — 276 judge calls in rapid succession — exceeded the provider's rate limit after roughly 80 calls.

### Design Decision
`metrics.py` enforces a 1-second minimum interval between judge calls, and results are cached for 24 hours so repeated eval runs are instant.

### Trade-off
Total wall-clock time for a full evaluation pass increased, in exchange for zero failed calls and free re-runs.

### Outcome
The evaluation suite completes reliably and can be re-run at no additional API cost.

---

> **Why BERTScore alone would have been misleading:** the untuned baseline scored 0.805 BERTScore — above the 0.70 target — despite 0% JSON validity, because embedding similarity rewards natural language that mentions the right cities and concepts regardless of structure. ROUGE-L (0.126 for baseline vs. 0.436 for `itinerai-ft`) correctly penalizes the format mismatch instead. This was treated as a methodology finding, not a bug, and is documented in `RESULTS.md` as a caution against relying on embedding-based metrics alone for structured-output tasks.

---

## Serving

### Challenge
The `mcp` library's transitive dependency on an older `starlette` version conflicts with the `starlette` ≥0.27 required by FastAPI 0.116+ when installed into an environment that already has an incompatible version present.

### Design Decision
All dependencies are declared in a single root `requirements.txt`, so pip's resolver settles on a compatible version set when installing into a clean environment.

### Trade-off
None when following the documented install path; the conflict only appears if `mcp` is pre-installed separately at an incompatible version.

### Outcome
A clean `pip install -r requirements.txt` in a fresh environment avoids the conflict entirely.

---

## Infrastructure & Repository Management

### Challenge
City coordinates and haversine distance logic were duplicated identically across `routing_server.py` and `hotels_server.py`.

### Design Decision
Both were centralized — city coordinates into `config.CITY_COORDS`, distance calculation into `utils/geo.py` — with both MCP servers importing from the shared source.

### Trade-off
None; this is a straightforward deduplication with no downside.

### Outcome
A single source of truth for geographic data used across the MCP layer.

---

### Challenge
Generated artifacts (88MB training JSONL, 37MB traces, 14MB synthetic data, plus GGUF model files) are large relative to a typical GitHub repository and add no unique value, since they are fully reproducible from the pipeline.

### Design Decision
`.gitignore` excludes `data/synthetic/`, `data/traces/`, `data/training/`, `data/seeds/`, and all GGUF files. Only `data/evals/` — the 92-case golden set, evaluation results, and charts, under 3MB total — is committed.

### Trade-off
Anyone cloning the repository must regenerate the large datasets to fully reproduce training, rather than downloading them directly.

### Outcome
The repository stays lightweight while still containing the artifacts that matter most for review: the evaluation evidence.

---

## Lessons Learned

1. **Match compute to sequence length before choosing hardware.** Sequence length, not model size alone, determines whether a free-tier GPU session is viable.
2. **Cache aggressively when tool calls scale faster than the task's actual entity space.** A 20-city network needing only ~40 unique upstream calls, versus thousands of raw calls, made every free-tier API sufficient.
3. **Budget-matching data generation strategies makes comparisons meaningful.** Holding cost constant ($4 per phase) isolates signal quality as the only variable between SFT and distillation.
4. **A single strong structural metric beats several plausible-looking soft metrics.** ROUGE-L caught a failure mode (0% JSON validity) that BERTScore missed entirely.
5. **Curriculum training can silently erase earlier-stage behavior.** When stage output distributions diverge sharply, later training can overwrite, not build on, earlier skills — this needs explicit safeguards, not just sequencing.
6. **Heuristic estimators are a legitimate substitute for unavailable ground truth, if disclosed.** A haversine-based cost model kept the project reproducible and free without misrepresenting its accuracy.
7. **Standardizing on a protocol (MCP) pays off beyond the immediate project.** Tools built for one agent chain worked unmodified in other MCP-compatible clients, with no extra integration cost.
8. **Idempotent, checkpointed pipelines are cheap insurance on paid API jobs.** Append-on-write output and ID-skip logic turned a multi-hour crash risk into a non-issue.
9. **Not every failure needs an immediate fix.** Documenting the curriculum JSON collapse as a finding — with a clear proposed remedy — was more valuable within scope than forcing a rushed patch.