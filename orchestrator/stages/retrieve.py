"""`retrieve` — one search per planned question (ARCH §5).

Every memory the system returned is stored, in the order it ranked them, so retrieval
metrics stay recomputable offline by a third party. Three fields per memory is the whole
record: content, score, source_ids. There is no envelope left to preserve — the adapter
already mapped whatever its system returned onto those, and the old duplicate `response`
blob had no readers.
"""

import asyncio
from pathlib import Path

from orchestrator.artifacts import JsonlArtifact
from orchestrator.client import AdapterClient
from orchestrator.plan_records import QuestionRecord, parse_plan_record
from orchestrator.runlog import get_logger
from orchestrator.stages.fanout import run_each

log = get_logger("retrieve")


async def run_retrieve(run_dir: Path, client: AdapterClient, top_k: int) -> int:
    """Retrieve for every not-yet-done question; return how many were written."""
    plan = JsonlArtifact(run_dir / "plan.jsonl")
    out = JsonlArtifact(run_dir / "retrieve.jsonl")
    done = out.read_ids()

    pending: list[QuestionRecord] = []
    for raw in plan.stream():
        record = parse_plan_record(raw)
        if isinstance(record, QuestionRecord) and record.id not in done:
            pending.append(record)

    if done:
        log.info("resuming: %d questions already retrieved", len(done))
    total = len(pending)
    lock = asyncio.Lock()
    completed = 0

    async def retrieve_one(record: QuestionRecord) -> None:
        nonlocal completed
        result = await client.search(record.user_id, record.question, top_k=top_k)
        async with lock:
            completed += 1
            # The query is content and stays out; how many memories came back does not.
            log.info(
                "search %d/%d id=%s memories=%d attempts=%d latency_ms=%d",
                completed,
                total,
                # The record id: `q_id` is unique only within one context.
                record.id,
                len(result.memories),
                result.attempts,
                result.latency_ms,
            )
            out.append(
                {
                    "id": record.id,
                    "user_id": record.user_id,
                    "query": record.question,
                    "top_k": top_k,
                    "attempts": result.attempts,
                    "latency_ms": result.latency_ms,
                    "started_at_ms": result.started_at_ms,
                    "ended_at_ms": result.ended_at_ms,
                    "memories": [
                        {
                            "content": memory.content,
                            "score": memory.score,
                            "source_ids": list(memory.source_ids),
                        }
                        for memory in result.memories
                    ],
                }
            )

    await run_each(pending, retrieve_one)
    return len(pending)
