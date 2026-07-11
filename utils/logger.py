"""
Structured JSON logger used by all phases.
Usage:
    from utils.logger import get_logger
    log = get_logger("phase1", "generation")
    log.info("Batch processed", batch_idx=3, valid=4, failed=1)
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }
        # Merge any extra keyword args passed via log.info("msg", key=val)
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _KwargsAdapter(logging.LoggerAdapter):
    """Lets callers write:  log.info("msg", key=val, other=val2)"""
    def process(self, msg, kwargs):
        extra = kwargs.pop("extra", {})
        # Every remaining kwarg is treated as a structured field
        extra["extra_fields"] = {k: v for k, v in kwargs.items()}
        kwargs.clear()
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(phase: str, name: str) -> _KwargsAdapter:
    """
    Returns a structured logger that writes JSON lines to:
      logs/<phase>/<name>.log   (file, always)
      stdout                    (only for WARNING+)
    """
    log_dir = Path(__file__).parent.parent / "logs" / phase
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"{phase}.{name}")
    if logger.handlers:
        return _KwargsAdapter(logger, {})

    logger.setLevel(logging.DEBUG)
    fmt = _JSONFormatter()

    # File handler — all levels
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler — WARNING+ only (keeps terminal readable during long runs)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.propagate = False
    return _KwargsAdapter(logger, {})
