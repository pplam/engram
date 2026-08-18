"""`ingest` — write every planned chunk into the system under test (ARCH §5).

Thin by design: read `plan.jsonl`, skip IDs already done, call the adapter, append one
record per chunk. A contract violation propagates and fails the run.
"""

import asyncio
from pathlib import Path

from contract.adapter import Message
from orchestrator.artifacts import JsonlArtifact
from orchestrator.client import AdapterClient
from orchestrator.plan_records import ChunkRecord, parse_plan_record
from orchestrator.runlog import get_logger
from orchestrator.stages.fanout import run_each

log = get_logger("ingest")


def _messages(record: ChunkRecord) -> list[Message]:
    """Return the chunk's messages as the typed records the adapter is handed."""
    return [
        Message(
            role=str(raw.get("role", "")),
            content=str(raw.get("content", "")),
            timestamp=raw.get("timestamp"),
        )
        for raw in record.messages
    ]


async def run_ingest(run_dir: Path, client: AdapterClient) -> int:
    """Ingest every not-yet-done chunk; return how many were written."""
    plan = JsonlArtifact(run_dir / "plan.jsonl")
    out = JsonlArtifact(run_dir / "ingest.jsonl")
    done = out.read_ids()

    pending: list[ChunkRecord] = []
    for raw in plan.stream():
        record = parse_plan_record(raw)
        if isinstance(record, ChunkRecord) and record.id not in done:
            pending.append(record)

    if done:
        log.info("resuming: %d chunks already ingested", len(done))
    total = len(pending)
    lock = asyncio.Lock()
    completed = 0

    async def ingest_one(record: ChunkRecord) -> None:
        nonlocal completed
        result = await client.add(record.user_id, _messages(record), chunk_id=record.chunk_id)
        async with lock:
            completed += 1
            # Under the lock so the counter matches what has actually been appended.
            log.info(
                "add %d/%d id=%s attempts=%d latency_ms=%d",
                completed,
                total,
                # The record id, not `chunk_id`: chunk ids restart per context, so two
                # contexts both have a `c0` and the log could not tell them apart.
                record.id,
                result.attempts,
                result.latency_ms,
            )
            out.append(
                {
                    "id": record.id,
                    "user_id": record.user_id,
                    "chunk_id": record.chunk_id,
                    "status": "ok",
                    "attempts": result.attempts,
                    "latency_ms": result.latency_ms,
                    "started_at_ms": result.started_at_ms,
                    "ended_at_ms": result.ended_at_ms,
                }
            )

    await run_each(pending, ingest_one)
    return len(pending)
