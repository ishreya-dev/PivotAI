"""
Shared constants for the ItinerAI-Bench project.
All phases import from here — never hardcode tiers, cities, or model names.
"""

# ─── Budget Tiers ────────────────────────────────────────────────────────────
# hotel_stars  → OSM stars tag / Overpass hotel quality filter (1–5)
# price_level  → OSM price_level tag (0–4)
# min_savings_pct → minimum savings% for a pivot to be valid (lower for cheap tiers)
BUDGET_TIERS: dict[str, dict] = {
    "Shoestring":   {"min_daily": 1500,  "max_daily": 3000,  "hotel_stars": 1, "price_level": 1, "min_savings_pct": 4.0},
    "Budget+":      {"min_daily": 3001,  "max_daily": 6000,  "hotel_stars": 2, "price_level": 2, "min_savings_pct": 4.0},
    "Mid-Range":    {"min_daily": 6001,  "max_daily": 12000, "hotel_stars": 3, "price_level": 2, "min_savings_pct": 5.0},
    "Premium":      {"min_daily": 12001, "max_daily": 25000, "hotel_stars": 4, "price_level": 3, "min_savings_pct": 5.0},
    "Ultra-Luxury": {"min_daily": 25001, "max_daily": 80000, "hotel_stars": 5, "price_level": 4, "min_savings_pct": 5.0},
}

VALIDATION_COST_MARGIN = 0.20   # ±20% tolerance on budget bounds

# Trip types where hostels/dormitories are never appropriate
NO_HOSTEL_TYPES: set[str] = {"Business", "Premium", "Ultra-Luxury"}

# ─── Cities ──────────────────────────────────────────────────────────────────
# Major metro hubs — common starting cities for domestic trips
HUB_CITIES: list[str] = [
    "Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai",
    "Pune", "Ahmedabad", "Kolkata",
]
# Leisure and tourist destinations
LEISURE_CITIES: list[str] = [
    "Goa", "Jaipur", "Varanasi", "Rishikesh", "Kochi",
    "Agra", "Shimla", "Darjeeling", "Mysore", "Udaipur",
    "Amritsar", "Guwahati",
]
ALL_CITIES: list[str] = HUB_CITIES + LEISURE_CITIES  # 20 cities total

# Fallback city-centre coordinates for all 20 ItinerAI-Bench cities.
# Used by MCP routing and hotels servers when Nominatim is unavailable.
CITY_COORDS: dict[str, tuple[float, float]] = {
    "Delhi":      (28.6139, 77.2090),
    "Mumbai":     (19.0760, 72.8777),
    "Bangalore":  (12.9716, 77.5946),
    "Hyderabad":  (17.3850, 78.4867),
    "Chennai":    (13.0827, 80.2707),
    "Pune":       (18.5204, 73.8567),
    "Ahmedabad":  (23.0225, 72.5714),
    "Kolkata":    (22.5726, 88.3639),
    "Goa":        (15.2993, 74.1240),
    "Jaipur":     (26.9124, 75.7873),
    "Varanasi":   (25.3176, 82.9739),
    "Rishikesh":  (30.0869, 78.2676),
    "Kochi":      (9.9312,  76.2673),
    "Agra":       (27.1767, 78.0081),
    "Shimla":     (31.1048, 77.1734),
    "Darjeeling": (27.0360, 88.2627),
    "Mysore":     (12.2958, 76.6394),
    "Udaipur":    (24.5854, 73.7125),
    "Amritsar":   (31.6340, 74.8723),
    "Guwahati":   (26.1445, 91.7362),
}

TRIP_TYPES: list[str] = ["Solo", "Family", "Group", "Couple", "Business"]

# 8 intents — expands the original 4 to cover food, nightlife, shopping, wildlife
# Business trips always keep "Business" as primary intent
INTENTS: list[str] = [
    "Adventure", "Relax", "Cultural", "Business",
    "Foodie", "Nightlife", "Shopping", "Wildlife",
]
NON_BUSINESS_INTENTS: list[str] = [i for i in INTENTS if i != "Business"]

# ─── Intent → OSM Tag Filters ────────────────────────────────────────────────
# Used by Concierge Agent's overpass_server queries
INTENT_OSM_TAGS: dict[str, list[tuple[str, str]]] = {
    "Adventure":  [("sport", "*"), ("leisure", "climbing"), ("leisure", "water_park"), ("natural", "peak")],
    "Relax":      [("leisure", "beach"), ("amenity", "spa"), ("leisure", "park"), ("leisure", "garden")],
    "Cultural":   [("tourism", "museum"), ("historic", "*"), ("amenity", "place_of_worship"), ("tourism", "attraction")],
    "Business":   [("amenity", "conference_centre"), ("office", "company"), ("amenity", "coworking_space")],
    "Foodie":     [("amenity", "restaurant"), ("amenity", "cafe"), ("amenity", "food_court")],
    "Nightlife":  [("amenity", "bar"), ("amenity", "nightclub"), ("amenity", "pub")],
    "Shopping":   [("shop", "mall"), ("shop", "department_store"), ("amenity", "marketplace")],
    "Wildlife":   [("leisure", "nature_reserve"), ("amenity", "zoo"), ("boundary", "protected_area"), ("tourism", "zoo")],
}

# ─── OpenRouteService Rate Limits (free tier) ────────────────────────────────
# Directions V2: 2000/day, 40/min  |  Geocoding: 3000/day, 100/min
# Caching collapses 500 runs → ~40 unique city pairs — well within limits.
ORS_DIRECTIONS_PER_MIN  = 40
ORS_GEOCODING_PER_MIN   = 100

# ─── MCP Server URLs (local development) ─────────────────────────────────────
MCP_SERVERS: dict[str, str] = {
    "routing":  "http://localhost:8001",
    "hotels":   "http://localhost:8002",
    "overpass": "http://localhost:8003",
    "search":   "http://localhost:8004",
}

# ─── Models ───────────────────────────────────────────────────────────────────
DEEPSEEK_AGENT_MODEL = "deepseek-chat"           # DeepSeek V4 Flash — Phase 2 agents + Phase 4 eval judge
DEEPSEEK_BASE_URL    = "https://api.deepseek.com"

# ── Active Phase 2 LLM provider ──────────────────────────────────────────────
# Switch this to change the LLM used by all three agents.
# "deepseek" → needs DEEPSEEK_API_KEY (paid — used deepseek-chat / DeepSeek V4 Flash)
# "groq"     → needs GROQ_API_KEY     (free tier, ~12 hrs for 500 records)
# "openai"   → needs OPENAI_API_KEY   ($1.71 for 500 records on gpt-4.1-nano)
PHASE2_PROVIDER = "deepseek"   # ← change this to switch providers
GEMINI_MODEL        = "gemini-2.0-flash"        # Phase 1 optional provider (not used by default)
GROQ_MODEL          = "llama-3.1-8b-instant"    # Phase 1 groq data gen
OPENAI_MODEL        = "gpt-4o-mini"             # Phase 1: primary data gen (paid, fast)
DEFAULT_PROVIDER    = "openai"                  # Phase 1 default provider
SLM_FT_MODEL         = "itinerai-ft"         # Ollama model name (fine-tuned)
SLM_DIST_MODEL       = "itinerai-distill"    # Ollama model name (distilled)
SLM_CURRICULUM_MODEL = "itinerai-curriculum" # Ollama model name (curriculum-trained)
SLM_BASELINE_MODEL   = "llama3.1:8b"         # Untuned base — establishes pre-training floor
OLLAMA_BASE_URL      = "http://localhost:11434"

# ─── Paths ────────────────────────────────────────────────────────────────────
import pathlib

ROOT = pathlib.Path(__file__).parent
DATA_DIR      = ROOT / "data"
SEEDS_DIR     = DATA_DIR / "seeds"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
TRACES_DIR    = DATA_DIR / "traces"
TRAINING_DIR  = DATA_DIR / "training"
EVALS_DIR     = DATA_DIR / "evals"
LOGS_DIR      = ROOT / "logs"
MODELS_DIR    = ROOT / "models"
FINETUNE_DIR   = MODELS_DIR / "finetune"
DISTILL_DIR    = MODELS_DIR / "distill"
CURRICULUM_DIR = MODELS_DIR / "curriculum"
CACHE_DIR     = ROOT / ".cache"