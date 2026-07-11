"""
Shared JSON extraction utility used by all three agents.
Uses json.JSONDecoder.raw_decode() to handle arbitrarily nested JSON — the
simple regex approach breaks at 3+ levels of nesting (optimizer output has 3).
"""

import json
from typing import Any


def extract_json(text: str) -> dict[str, Any]:
    """
    Find and return the largest valid JSON object in text.
    Scans every '{' position and tries to parse outward.
    Returns the object with the greatest character span (most complete JSON).
    """
    decoder = json.JSONDecoder()
    best: dict[str, Any] = {}
    best_span = 0

    for i, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, end_pos = decoder.raw_decode(text, i)
            if isinstance(obj, dict):
                span = end_pos - i
                if span > best_span:
                    best = obj
                    best_span = span
        except json.JSONDecodeError:
            continue

    return best