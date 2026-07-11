"""
pivotai MCP Routing Server — port 8001
Wraps OpenRouteService (road distances) + Nominatim (geocoding).
No auth required for Nominatim. ORS_API_KEY needed for precise road routing.
"""

import os
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MCP_SERVERS, CITY_COORDS
from utils.cache import api_cache
from utils.geo import haversine_km
from utils.logger import get_logger

load_dotenv()
log = get_logger("phase2", "mcp_servers")
_PORT = int(MCP_SERVERS["routing"].split(":")[-1])
mcp = FastMCP("pivotai-routing", host="0.0.0.0", port=_PORT)

_ORS_KEY = os.getenv("ORS_API_KEY", "")


@api_cache(ttl=86400 * 30)
def _geocode_cached(city: str) -> dict[str, Any]:
    try:
        resp = httpx.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{city}, India", "format": "json", "limit": 1},
            headers={"User-Agent": "pivotai/1.0 (portfolio project)"},
            timeout=10.0,
        )
        resp.raise_for_status()
        results = resp.json()
        if results:
            return {
                "city": city,
                "lat": float(results[0]["lat"]),
                "lng": float(results[0]["lon"]),
                "source": "nominatim",
            }
    except Exception as exc:
        log.warning("Nominatim geocode failed", city=city, error=str(exc))

    if city in CITY_COORDS:
        lat, lng = CITY_COORDS[city]
        return {"city": city, "lat": lat, "lng": lng, "source": "hardcoded"}

    return {"city": city, "lat": None, "lng": None, "error": f"Unknown city: {city}"}


@api_cache(ttl=86400 * 7)
def _route_cached(origin: str, dest: str, mode: str) -> dict[str, Any]:
    og = _geocode_cached(origin)
    dg = _geocode_cached(dest)
    if not og.get("lat") or not dg.get("lat"):
        return {"error": f"Cannot geocode {origin} or {dest}"}

    straight_km = haversine_km(og["lat"], og["lng"], dg["lat"], dg["lng"])
    road_km = straight_km * 1.35  # default road factor; overridden by ORS below

    flight_dur = round(straight_km / 800 + 2.5, 1)   # flight + 2.5h airport time
    flight_cost = int(straight_km * 3 + 1500)          # ₹3/km + ₹1500 base

    if mode == "flight":
        return {
            "origin": origin, "dest": dest, "mode": "flight",
            "distance_km": round(straight_km),
            "duration_hours": flight_dur,
            "cost_per_person_inr": flight_cost,
            "transit_options": [
                {"mode": "flight", "duration_hours": flight_dur,
                 "cost_per_person_inr": flight_cost, "frequency": "multiple daily"},
            ],
        }

    # Try ORS for precise road distance (car/bus/train all use road network as proxy)
    # Rate limit: 40 requests/min → 1.5s gap between calls is safe
    if _ORS_KEY:
        try:
            import time
            import openrouteservice
            time.sleep(1.5)  # respect 40 req/min free-tier limit
            client = openrouteservice.Client(key=_ORS_KEY)
            coords = [(og["lng"], og["lat"]), (dg["lng"], dg["lat"])]
            r = client.directions(coords, profile="driving-car", format="geojson")
            road_km = r["features"][0]["properties"]["segments"][0]["distance"] / 1000
        except Exception as exc:
            log.warning("ORS routing failed, using heuristic", error=str(exc))

    _speeds    = {"train": 80,  "bus": 55,  "car": 70}
    _cpkm      = {"train": 1.2, "bus": 0.8, "car": 2.5}
    _base      = {"train": 200, "bus": 100, "car": 0}

    speed = _speeds.get(mode, 80)
    dur   = round(road_km / speed, 1)
    cost  = int(road_km * _cpkm.get(mode, 1.2) + _base.get(mode, 0))

    # Always show transit alternatives so the agent can compare
    transit_options: list[dict] = [
        {"mode": mode, "duration_hours": dur, "cost_per_person_inr": cost, "frequency": "multiple daily"},
    ]
    if straight_km > 600:  # long haul: add flight alternative
        transit_options.append(
            {"mode": "flight", "duration_hours": flight_dur,
             "cost_per_person_inr": flight_cost, "frequency": "multiple daily"}
        )
    elif mode != "train":  # short haul: show train if we're not already on it
        train_cost = int(road_km * 1.2 + 200)
        transit_options.append(
            {"mode": "train", "duration_hours": round(road_km / 80, 1),
             "cost_per_person_inr": train_cost, "frequency": "multiple daily"}
        )

    return {
        "origin": origin, "dest": dest, "mode": mode,
        "distance_km": round(road_km),
        "duration_hours": dur,
        "cost_per_person_inr": cost,
        "transit_options": transit_options,
    }


@mcp.tool()
def geocode_city(city_name: str) -> dict:
    """
    Return latitude/longitude for an Indian city.
    Falls back to hardcoded coordinates if Nominatim is unavailable.
    """
    result = _geocode_cached(city_name)
    log.info("geocode_city", city=city_name, source=result.get("source", "unknown"))
    return result


@mcp.tool()
def get_route(origin_city: str, dest_city: str, mode: str = "train") -> dict:
    """
    Get travel distance, duration, and per-person cost between two Indian cities.
    Returns transit_options list with alternatives (e.g. train + flight for long routes).
    mode: 'flight' | 'train' | 'bus' | 'car'
    """
    result = _route_cached(origin_city, dest_city, mode)
    log.info(
        "get_route",
        origin=origin_city, dest=dest_city, mode=mode,
        distance_km=result.get("distance_km"),
        cost_inr=result.get("cost_per_person_inr"),
    )
    return result


if __name__ == "__main__":
    log.info("Starting pivotai routing MCP server", port=_PORT, ors_key_set=bool(_ORS_KEY))
    mcp.run(transport="sse")
