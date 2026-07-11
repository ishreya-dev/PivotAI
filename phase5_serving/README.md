# Phase 5 — Inference API

REST API that wraps all four pivotai models via Ollama. Accepts a traveler persona and returns an optimized itinerary from any of the three fine-tuned models or the untuned baseline.

---

## Setup

```bash
# From the project root — all dependencies are in the central requirements.txt
pip install -r requirements.txt

# Ollama must be running with models loaded
ollama serve
ollama create pivotai-ft         -f phase3_training/notebooks/modelfiles/Modelfile.ft
ollama create pivotai-distill    -f phase3_training/notebooks/modelfiles/Modelfile.distill
ollama create pivotai-curriculum -f phase3_training/notebooks/modelfiles/Modelfile.curriculum
ollama pull llama3.1:8b
```

---

## Start the Server

```bash
uvicorn phase5_serving.api.main:app --reload --port 8000
```

Interactive Swagger docs: **http://localhost:8000/docs**

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Ollama reachability + which models are loaded |
| `GET` | `/models` | All 4 registered models with training descriptions |
| `POST` | `/optimize` | Run inference — returns structured itinerary + raw output + latency |
| `GET` | `/results/summary` | Latest Phase 4 eval summary JSON (instant, no inference) |
| `GET` | `/results/compare` | Head-to-head win rates from eval summary |

---

## POST /optimize

**Request**

```json
{
  "model": "pivotai-ft",
  "persona": {
    "starting_city": "Mumbai",
    "destination_city": "Delhi",
    "type": "Solo",
    "size": {"adults": 1, "children": 0},
    "intents": ["Adventure"],
    "budget": "Shoestring",
    "duration_days": 5,
    "duration_nights": 4
  }
}
```

**Response**

```json
{
  "model": "pivotai-ft",
  "output": { "optimized": {...}, "pivot_analysis": "..." },
  "raw_output": "...",
  "latency_ms": 42180
}
```

Valid model names: `pivotai-ft`, `pivotai-distill`, `pivotai-curriculum`, `llama3.1:8b`.  
Any other value returns a 422 with a list of valid options — model names are validated against a registry to prevent injection.

---

## Example cURL

```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pivotai-ft",
    "persona": {
      "starting_city": "Mumbai",
      "destination_city": "Goa",
      "type": "Couple",
      "size": {"adults": 2, "children": 0},
      "intents": ["Relax", "Foodie"],
      "budget": "Mid-Range",
      "duration_days": 4,
      "duration_nights": 3
    }
  }'
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_TIMEOUT` | `600` | Seconds before Ollama call times out (CPU inference: 30–120s per query) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address (set in `config.py`) |

---

## Design Notes

- `async def` endpoints + `httpx.AsyncClient` — non-blocking while Ollama generates; server stays responsive to health checks during long inference
- Model names validated against `MODEL_REGISTRY` in `schemas.py` — prevents passing arbitrary strings to Ollama
- `/results/summary` and `/results/compare` serve pre-computed eval data instantly — no Ollama call needed
- Prompt format is identical to the eval pipeline (`build_prompt` from `phase4_evals/utils.py`) — inference and evaluation use the same Alpaca template
