"""
Phase 2 main entrypoint.

Features:
  - Auto-resume: scans data/traces/ for already-processed record IDs — no --offset needed
  - Parallel processing: --concurrency N workers (default 5) with shared rate limiter
  - Quota-safe: stops cleanly on daily limit, prints exact command to resume tomorrow

Usage:
    python phase2_agents/run.py                        # run all 5000, auto-resume
    python phase2_agents/run.py --limit 200            # cap at 200 records
    python phase2_agents/run.py --concurrency 8        # more parallel workers
    python phase2_agents/run.py --record-id <uuid>     # debug a specific record
    python phase2_agents/run.py --fresh                # ignore existing traces, start over
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from config import MCP_SERVERS, PHASE2_PROVIDER, SYNTHETIC_DIR, TRACES_DIR
from phase2_agents.llm_utils import QuotaDailyExhausted
from phase2_agents.mcp_adapter import MCPAdapter
from phase2_agents.supervisor import run_trace
from utils.logger import get_logger

log = get_logger("phase2", "progress")


# ─── Auto-resume: find already-processed record IDs ──────────────────────────

def _get_processed_ids(traces_dir: Path) -> set[str]:
    """Scan all existing trace files and return the set of phase1_record_ids done."""
    processed: set[str] = set()
    for trace_file in sorted(traces_dir.glob("agent_traces_*.jsonl")):
        try:
            with open(trace_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    pid = data.get("phase1_record_id") or data.get("id")
                    if pid:
                        processed.add(pid)
        except Exception:
            pass
    return processed


# ─── Dataset loader ───────────────────────────────────────────────────────────

def _load_records(
    dataset_path: Path,
    limit: int,
    record_id: str | None,
    processed_ids: set[str],
    fresh: bool,
) -> list[dict]:
    all_records: list[dict] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if record_id:
        matches = [r for r in all_records if r.get("id") == record_id]
        if not matches:
            raise ValueError(f"Record ID not found: {record_id}")
        return matches

    # Filter out already-processed records (auto-resume)
    if not fresh and processed_ids:
        pending = [r for r in all_records if r.get("id") not in processed_ids]
    else:
        pending = all_records

    return pending[:limit]


# ─── Per-record worker ────────────────────────────────────────────────────────

async def _process_single(
    record: dict,
    output_path: Path,
    file_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
    counters: dict,
    verbose: bool,
    total: int,
    stop_event: asyncio.Event,
    adapter: "MCPAdapter",
) -> None:
    if stop_event.is_set():
        return

    persona = record.get("persona", {})
    origin  = persona.get("starting_city", "?")
    dest    = persona.get("destination_city", "?")
    record_id = record.get("id", "unknown")

    async with semaphore:
        if stop_event.is_set():
            return

        t0 = time.monotonic()
        try:
            trace   = await run_trace(record, adapter=adapter)   # reuse shared adapter
            elapsed = time.monotonic() - t0
            line    = json.dumps(trace.model_dump(), ensure_ascii=False, default=str)

            async with file_lock:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                counters["success"] += 1
                counters["savings"]  += trace.savings_pct
                done = counters["success"] + counters["fail"]

            if verbose:
                avg = counters["savings"] / counters["success"]
                print(
                    f"[{done}/{total}] ✓ {origin} → {dest} | "
                    f"{trace.savings_pct:.1f}% savings | "
                    f"{trace.grounding.total_api_calls} API calls | "
                    f"{elapsed:.1f}s  (avg {avg:.1f}%)"
                )

            log.info("Record complete", record_id=record_id,
                     savings_pct=trace.savings_pct, elapsed_s=round(elapsed, 1))

        except QuotaDailyExhausted as exc:
            stop_event.set()
            async with file_lock:
                counters["fail"] += 1
            msg = str(exc)
            if "credits" in msg or "billing" in msg:
                print(f"\n⚠  API account out of credits after {counters['success']} successful records.")
                print(f"   Check your API key balance and top up — then re-run.")
                print(f"   Auto-resume will skip completed records.\n")
            else:
                print(f"\n⚠  Daily quota exhausted after {counters['success']} records.")
                print(f"   Run again tomorrow — auto-resume will skip completed records.\n")
            log.warning("Quota/billing stop", records_done=counters["success"], error=msg)

        except Exception as exc:
            elapsed = time.monotonic() - t0
            async with file_lock:
                counters["fail"] += 1
                done = counters["success"] + counters["fail"]
            log.warning("Record failed", record_id=record_id,
                        error=str(exc), elapsed_s=round(elapsed, 1))
            if verbose:
                print(f"[{done}/{total}] ✗ {origin} → {dest} — {str(exc)[:80]}")


# ─── Parallel batch runner ────────────────────────────────────────────────────

async def _run_parallel(
    records: list[dict],
    output_path: Path,
    verbose: bool,
    concurrency: int,
) -> tuple[int, int, float]:
    semaphore  = asyncio.Semaphore(concurrency)
    file_lock  = asyncio.Lock()
    stop_event = asyncio.Event()
    counters   = {"success": 0, "fail": 0, "savings": 0.0}

    # ONE shared adapter = ONE set of 4 SSE connections for all workers.
    # Prevents file-descriptor exhaustion from N workers × 4 connections each.
    async with MCPAdapter(MCP_SERVERS) as shared_adapter:
        tasks = [
            _process_single(
                record, output_path, file_lock, semaphore,
                counters, verbose, len(records), stop_event,
                adapter=shared_adapter,
            )
            for record in records
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    avg = counters["savings"] / counters["success"] if counters["success"] else 0.0
    return counters["success"], counters["fail"], avg


# ─── CLI entrypoint ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ItinerAI-Bench Phase 2 — agent trace generator")
    parser.add_argument("--limit",       type=int, default=5000, help="Max pending records to process")
    parser.add_argument("--concurrency", type=int, default=5,    help="Parallel workers (default 5)")
    parser.add_argument("--record-id",   type=str, default="",   help="Process a single record by UUID")
    parser.add_argument("--dataset",     type=str, default="",   help="Path to Phase 1 JSONL (auto-detects latest)")
    parser.add_argument("--fresh",       action="store_true",    help="Ignore existing traces, start over")
    parser.add_argument("--verbose",     action="store_true",    help="Print per-record progress")
    args = parser.parse_args()

    # Locate dataset
    if args.dataset:
        dataset_path = Path(args.dataset)
    else:
        candidates = sorted(SYNTHETIC_DIR.glob("*.jsonl"))
        if not candidates:
            print("ERROR: No Phase 1 JSONL found in data/synthetic/")
            sys.exit(1)
        dataset_path = candidates[-1]

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    key_map = {"groq": "GROQ_API_KEY", "deepseek": "DEEPSEEK_API_KEY", "openai": "OPENAI_API_KEY"}
    required_key = key_map.get(PHASE2_PROVIDER.lower())
    if required_key and not os.getenv(required_key):
        print(f"ERROR: {required_key} not set in .env (PHASE2_PROVIDER={PHASE2_PROVIDER})")
        sys.exit(1)

    # Auto-resume: find already-processed IDs
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    processed_ids = set() if args.fresh else _get_processed_ids(TRACES_DIR)

    # Load pending records
    records = _load_records(
        dataset_path,
        limit=args.limit,
        record_id=args.record_id or None,
        processed_ids=processed_ids,
        fresh=args.fresh,
    )

    if not records:
        print("✓ All records already processed. Nothing to do.")
        sys.exit(0)

    # Output file for this session
    timestamp    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path  = TRACES_DIR / f"agent_traces_{timestamp}.jsonl"

    already_done = len(processed_ids)
    print(f"ItinerAI-Bench Phase 2 — Agent Trace Generator")
    print(f"  Dataset:        {dataset_path.name}")
    print(f"  Already done:   {already_done} records (auto-resumed)")
    print(f"  Pending:        {len(records)} records")
    print(f"  Concurrency:    {args.concurrency} parallel workers")
    print(f"  Output:         {output_path}")
    print()

    log.info("Run start", dataset=str(dataset_path), pending=len(records),
             already_done=already_done, concurrency=args.concurrency,
             output=str(output_path))

    t_start = time.monotonic()
    success, fail, avg_savings = asyncio.run(
        _run_parallel(records, output_path, args.verbose, args.concurrency)
    )
    total_elapsed = time.monotonic() - t_start

    print()
    print(f"── Phase 2 Complete ──────────────────────────────")
    print(f"  Processed:   {success + fail} records this session")
    print(f"  Successful:  {success}")
    print(f"  Failed:      {fail}")
    print(f"  Avg savings: {avg_savings:.1f}%")
    print(f"  Total time:  {total_elapsed / 60:.1f} min")
    print(f"  Output:      {output_path}")
    total_done = already_done + success
    print(f"  Grand total: {total_done} / {already_done + len(records)} records done")

    log.info("Run complete", success=success, fail=fail,
             avg_savings_pct=round(avg_savings, 2),
             total_elapsed_min=round(total_elapsed / 60, 1),
             grand_total_done=total_done)


if __name__ == "__main__":
    main()