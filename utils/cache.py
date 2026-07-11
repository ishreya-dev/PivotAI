"""
Disk-based API response cache shared by all MCP servers and agents.

Why: Our 10-city universe produces ~90 unique (origin, destination) routing pairs.
With caching, 1000 agent runs collapse to ~90 real ORS API calls — well inside the
2000/day free limit. Same logic applies to hotel and POI lookups.

Usage:
    from utils.cache import api_cache

    @api_cache(ttl=86400)  # 24 hours
    def get_route(origin: str, dest: str, mode: str) -> dict:
        ...  # real API call only on cache miss
"""

import functools
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import diskcache

_CACHE_DIR = Path(__file__).parent.parent / ".cache" / "api_responses"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_disk_cache = diskcache.Cache(str(_CACHE_DIR))


def _make_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Stable hash key from function name + arguments."""
    payload = json.dumps({"fn": func_name, "args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def api_cache(ttl: int = 86400):
    """
    Decorator that caches a function's return value to disk.
    ttl: cache lifetime in seconds (default 24 hours)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            key = _make_key(func.__qualname__, args, kwargs)
            cached = _disk_cache.get(key)
            if cached is not None:
                return cached
            result = func(*args, **kwargs)
            _disk_cache.set(key, result, expire=ttl)
            return result
        return wrapper
    return decorator


def cache_stats() -> dict:
    """Return hit/miss statistics for logging."""
    return {
        "size_bytes": _disk_cache.volume(),
        "item_count": len(_disk_cache),
        "directory":  str(_CACHE_DIR),
    }


def clear_cache() -> None:
    _disk_cache.clear()
