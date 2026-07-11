"""
pivotai MCP Overpass Server — port 8003
Wraps Overpass API (OpenStreetMap) for POI and restaurant lookups.
No API key required — Overpass is free and unlimited.
"""

import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BUDGET_TIERS, INTENT_OSM_TAGS, MCP_SERVERS
from utils.cache import api_cache
from utils.logger import get_logger

load_dotenv()
log = get_logger("phase2", "mcp_servers")
_PORT = int(MCP_SERVERS["overpass"].split(":")[-1])
mcp = FastMCP("pivotai-overpass", host="0.0.0.0", port=_PORT)

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Rough bounding boxes (lat_min, lng_min, lat_max, lng_max) for each city
# Used to scope Overpass queries to the right metro area
_CITY_BBOX: dict[str, tuple[float, float, float, float]] = {
    "Delhi":      (28.40, 76.84, 28.88, 77.35),
    "Mumbai":     (18.89, 72.77, 19.27, 72.99),
    "Bangalore":  (12.83, 77.46, 13.14, 77.75),
    "Hyderabad":  (17.20, 78.30, 17.55, 78.62),
    "Chennai":    (12.90, 80.14, 13.22, 80.31),
    "Pune":       (18.42, 73.75, 18.62, 73.95),
    "Ahmedabad":  (22.93, 72.49, 23.10, 72.66),
    "Kolkata":    (22.43, 88.25, 22.70, 88.44),
    "Goa":        (15.27, 73.87, 15.55, 74.13),
    "Jaipur":     (26.79, 75.68, 27.03, 75.90),
    "Varanasi":   (25.26, 82.88, 25.38, 83.05),
    "Rishikesh":  (30.04, 78.22, 30.14, 78.33),
    "Kochi":      (9.89,  76.21, 10.03, 76.35),
    "Agra":       (27.10, 77.94, 27.24, 78.09),
    "Shimla":     (31.06, 77.07, 31.15, 77.22),
    "Darjeeling": (26.99, 88.23, 27.07, 88.31),
    "Mysore":     (12.25, 76.57, 12.37, 76.70),
    "Udaipur":    (24.54, 73.65, 24.63, 73.79),
    "Amritsar":   (31.59, 74.82, 31.69, 74.94),
    "Guwahati":   (26.10, 91.67, 26.22, 91.82),
}


def _build_poi_query(bbox: tuple, tags: list[tuple[str, str]], limit: int) -> str:
    """Build Overpass QL query for given bbox and OSM tag filters."""
    lat_min, lng_min, lat_max, lng_max = bbox
    bbox_str = f"{lat_min},{lng_min},{lat_max},{lng_max}"
    filters = []
    for key, value in tags:
        val_filter = f'["{key}"]' if value == "*" else f'["{key}"="{value}"]'
        filters.append(f'node{val_filter}({bbox_str});')
        filters.append(f'way{val_filter}({bbox_str});')
    body = "\n  ".join(filters)
    return f'[out:json][timeout:25];\n(\n  {body}\n);\nout body {limit};'


@api_cache(ttl=86400)
def _overpass_query(query: str) -> list[dict]:
    """Execute an Overpass QL query and return the elements list."""
    try:
        resp = httpx.post(
            _OVERPASS_URL,
            data={"data": query},
            timeout=30.0,
            headers={"User-Agent": "pivotai/1.0"},
        )
        resp.raise_for_status()
        return resp.json().get("elements", [])
    except Exception as exc:
        log.warning("Overpass query failed", error=str(exc))
        return []


def _extract_poi(element: dict) -> dict[str, Any]:
    tags = element.get("tags", {})
    return {
        "name":     tags.get("name", "Unnamed"),
        "type":     tags.get("tourism") or tags.get("amenity") or tags.get("leisure") or tags.get("shop") or "poi",
        "address":  tags.get("addr:full") or tags.get("addr:street", ""),
        "lat":      element.get("lat") or (element.get("center", {}) or {}).get("lat"),
        "lng":      element.get("lon") or (element.get("center", {}) or {}).get("lon"),
        "opening_hours": tags.get("opening_hours", ""),
        "website":  tags.get("website", ""),
        "price_level": tags.get("price_level", ""),
    }


@mcp.tool()
def search_pois(city: str, intent: str, price_level: int = 2, limit: int = 10) -> dict:
    """
    Find points of interest in a city matching a traveler intent.
    intent: 'Adventure' | 'Relax' | 'Cultural' | 'Business' | 'Foodie' | 'Nightlife' | 'Shopping' | 'Wildlife'
    price_level: 1 (budget) to 4 (luxury) — maps to config.BUDGET_TIERS price_level
    Returns up to `limit` POIs with name, type, location, and opening hours.
    """
    bbox = _CITY_BBOX.get(city)
    tags = INTENT_OSM_TAGS.get(intent, [])

    if not bbox:
        return {"error": f"Unknown city: {city}", "pois": []}
    if not tags:
        return {"error": f"Unknown intent: {intent}", "pois": []}

    query = _build_poi_query(bbox, tags, limit * 2)
    elements = _overpass_query(query)

    pois = [_extract_poi(e) for e in elements if e.get("tags", {}).get("name")]
    pois = pois[:limit]

    log.info("search_pois", city=city, intent=intent, found=len(pois))
    return {"city": city, "intent": intent, "price_level": price_level, "pois": pois}


@mcp.tool()
def search_restaurants(city: str, price_level: int = 2, cuisine: str = "", limit: int = 10) -> dict:
    """
    Find restaurants in a city filtered by price level and optional cuisine.
    price_level: 1 (budget) to 4 (luxury)
    cuisine: optional filter e.g. 'Indian', 'Chinese', 'Italian' (empty = all)
    Returns up to `limit` restaurants with name, cuisine, price level, and location.
    """
    bbox = _CITY_BBOX.get(city)
    if not bbox:
        return {"error": f"Unknown city: {city}", "restaurants": []}

    lat_min, lng_min, lat_max, lng_max = bbox
    bbox_str = f"{lat_min},{lng_min},{lat_max},{lng_max}"

    cuisine_filter = f'["cuisine"="{cuisine}"]' if cuisine else ""
    query = (
        f'[out:json][timeout:25];\n'
        f'(\n'
        f'  node["amenity"="restaurant"]{cuisine_filter}({bbox_str});\n'
        f'  way["amenity"="restaurant"]{cuisine_filter}({bbox_str});\n'
        f'  node["amenity"="cafe"]{cuisine_filter}({bbox_str});\n'
        f');\n'
        f'out body {limit * 2};'
    )

    elements = _overpass_query(query)

    restaurants = []
    for e in elements:
        tags = e.get("tags", {})
        if not tags.get("name"):
            continue
        osm_price = tags.get("price_level", "")
        # Simple price heuristic: include if no price tag, or tag roughly matches
        restaurants.append({
            "name":         tags.get("name", "Unnamed"),
            "cuisine":      tags.get("cuisine", "Indian"),
            "amenity":      tags.get("amenity", "restaurant"),
            "price_level":  osm_price or str(price_level),
            "address":      tags.get("addr:street", ""),
            "lat":          e.get("lat") or (e.get("center", {}) or {}).get("lat"),
            "lng":          e.get("lon") or (e.get("center", {}) or {}).get("lon"),
            "opening_hours": tags.get("opening_hours", ""),
        })

    restaurants = restaurants[:limit]
    log.info("search_restaurants", city=city, price_level=price_level, cuisine=cuisine, found=len(restaurants))
    return {"city": city, "price_level": price_level, "cuisine": cuisine, "restaurants": restaurants}


if __name__ == "__main__":
    log.info("Starting pivotai overpass MCP server", port=_PORT)
    mcp.run(transport="sse")
