"""
LLM API helpers — async wrapper around any OpenAI-compatible client (DeepSeek, Groq).
Used by all Phase 2 agents.

llm_generate() — async wrapper with shared rate limiting, retry on 429/503, and
                 daily quota detection. Works with any OpenAI-compatible API.
msg_to_dict()  — converts an SDK message object to a plain dict for message history.
"""

import asyncio
import re
import time as _time
from typing import Any

from utils.logger import get_logger

log = get_logger("phase2", "agents")

# ─── Error markers ────────────────────────────────────────────────────────────
# insufficient_quota / insufficient_balance = account has no credits (never retryable)
# PerDay / daily = per-day quota exhausted (reset at midnight)
_BILLING_MARKERS   = ("insufficient_quota", "Insufficient Balance", "insufficient_balance", "402")
_DAILY_MARKERS     = ("PerDay", "per_day", "daily", "GenerateRequestsPerDayPerProjectPerModel")
_PER_MIN_MARKERS   = ("503", "UNAVAILABLE", "overloaded", "PerMinute", "per_minute",
                      "rate_limit", "Rate limit")


class QuotaDailyExhausted(RuntimeError):
    """Daily API quota used up — stop the run and resume tomorrow."""


def _parse_retry_delay(err_text: str) -> float:
    m = re.search(r"retry.*?(\d+)s", err_text, re.IGNORECASE)
    return float(m.group(1)) if m else 30.0


# ─── Shared async rate limiter ────────────────────────────────────────────────
# Lazy asyncio.Lock so it's safe to instantiate at module level.

class _AsyncRateLimiter:
    def __init__(self, calls_per_minute: float):
        self._interval = 60.0 / calls_per_minute
        self._lock: asyncio.Lock | None = None
        self._last: float = 0.0

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self) -> None:
        async with self._get_lock():
            gap = _time.monotonic() - self._last
            if gap < self._interval:
                await asyncio.sleep(self._interval - gap)
            self._last = _time.monotonic()


_rate_limiter_fast = _AsyncRateLimiter(calls_per_minute=400)  # DeepSeek: no throttle needed
_rate_limiter_groq = _AsyncRateLimiter(calls_per_minute=5)    # Groq free: 30K TPM ÷ ~5000 tok/call
_rate_limiter = _rate_limiter_fast  # default; switched per-provider in llm_client.py


# ─── OpenAI-compatible generate ───────────────────────────────────────────────

def _fake_text_response(text: str) -> Any:
    """Return a minimal response object containing plain text — no tool calls."""
    from types import SimpleNamespace
    msg = SimpleNamespace(role="assistant", content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


async def llm_generate(
    client,
    model: str,
    messages: list,
    tools: list | None = None,
    system: str = "",
    max_retries: int = 4,
) -> Any:
    """
    Async wrapper around client.chat.completions.create with:
    - Shared rate limiting (400 RPM target, safe with 5–10 concurrent workers)
    - Per-minute 429 / 503 retry with backoff
    - Daily quota → raises QuotaDailyExhausted
    - Phantom tool call (Llama) → extracts JSON and returns as text
    """
    full_messages = ([{"role": "system", "content": system}] if system else []) + messages

    for attempt in range(max_retries):
        await _rate_limiter.acquire()

        try:
            kwargs: dict = {"model": model, "messages": full_messages, "temperature": 0.3}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            return await asyncio.to_thread(client.chat.completions.create, **kwargs)

        except Exception as exc:
            err = str(exc)

            # Some models invent a fake return-tool instead of outputting text.
            # Extract the embedded JSON and surface it as a plain text response.
            if "tool_use_failed" in err:
                m = re.search(r"<function=\w+>(\{.+?\})</function>", err, re.DOTALL)
                if not m:
                    m = re.search(r"failed_generation.*?(\{.+?\})", err, re.DOTALL)
                if m:
                    log.warning("Phantom tool call — extracting JSON as text response")
                    return _fake_text_response(m.group(1))
                raise

            if any(marker in err for marker in _BILLING_MARKERS):
                raise QuotaDailyExhausted(
                    f"API account out of credits for '{model}'. "
                    f"Check your API key balance and top up — then re-run."
                ) from exc

            if any(marker in err for marker in _DAILY_MARKERS):
                raise QuotaDailyExhausted(
                    f"Daily quota exhausted for '{model}'. Resume tomorrow."
                ) from exc

            if any(marker in err for marker in _PER_MIN_MARKERS):
                wait = max(_parse_retry_delay(err), 5 * (2 ** attempt))
                log.warning("Rate-limit — retrying", attempt=attempt + 1,
                            wait_s=wait, error=err[:100])
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait)
                    continue

            raise

    raise RuntimeError(f"llm_generate: exceeded {max_retries} retries")


def msg_to_dict(msg: Any) -> dict:
    """Convert an SDK message object to a plain dict for message history."""
    d: dict = {"role": msg.role, "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return d