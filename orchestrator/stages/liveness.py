"""The gate between `ingest` and `retrieve`: did the writes land anywhere (ARCH §8).

`add` returning 200 is not evidence that anything was stored. A memory system whose
extraction step yields nothing — because a completion budget was too small, a prompt was
rejected, or a model returned something unparseable — accepts every chunk, stores none of
it, and reports success. The run then measures an empty store and publishes the result as
retrieval misses, which reads as a system that retrieves poorly rather than as an
integration that never wrote.

So this asks the store one question it should be able to answer and fails the run when it
answers with nothing. Per CLAUDE.md, a broken integration is not a score of zero.

Two deliberate limits. The probe is a **real planned question**, because a store that
filters by score threshold may legitimately return nothing for a synthetic string, and a
gate that fires on a healthy system is worse than no gate. And it probes a **bounded
sample** of contexts, because LongMemEval-S carries one context per question and probing
every one would cost a second retrieve pass to catch a misconfiguration that is global by
nature.

This does not detect *partial* loss — a store holding one chunk of twenty answers the
probe fine. Coverage across everything retrieved is computable offline from
`retrieve.jsonl` once `retrieve` has run, which is where it belongs: free, exact, and
over every query rather than a sample.
"""

from pathlib import Path

from contract.adapter import ContractViolation
from orchestrator.artifacts import JsonlArtifact
from orchestrator.client import AdapterClient
from orchestrator.plan_records import ChunkRecord, QuestionRecord, parse_plan_record
from orchestrator.runlog import get_logger

log = get_logger("liveness")

# How many contexts to probe. An empty store is a property of the integration rather than
# of one context, so a handful is as diagnostic as all of them and costs a handful of
# searches instead of a second retrieve pass.
DEFAULT_SAMPLE = 5


async def run_liveness(
    run_dir: Path,
    client: AdapterClient,
    top_k: int,
    sample: int = DEFAULT_SAMPLE,
    stores_nothing: bool = False,
    ingested: int | None = None,
) -> int:
    """Probe up to `sample` ingested contexts; return how many answered. Raises if any is empty."""
    if stores_nothing:
        # The `no_memory` floor: an empty store is what this system is measuring, so
        # there is nothing here to catch.
        log.info("liveness skipped: system declares it stores nothing")
        return 0

    if ingested == 0:
        # A fully resumed run: `ingest` skips chunks already recorded, so this invocation
        # wrote nothing and the store's contents were settled by the earlier one. The gate
        # vouches for writes it saw happen, and there were none.
        log.info("liveness skipped: this invocation wrote no chunks")
        return 0

    written: dict[str, int] = {}
    probes: dict[str, str] = {}

    for raw in JsonlArtifact(run_dir / "plan.jsonl").stream():
        record = parse_plan_record(raw)
        if isinstance(record, ChunkRecord):
            written[record.user_id] = written.get(record.user_id, 0) + 1
        elif isinstance(record, QuestionRecord):
            # First question per context: the plan is ordered, so which one is chosen is
            # deterministic and a re-run probes the same thing.
            probes.setdefault(record.user_id, record.question)

    # Only contexts that received writes and have a question to ask about them.
    candidates = [uid for uid in written if uid in probes][:sample]
    if not candidates:
        log.info("liveness skipped: no ingested context carries a question to probe with")
        return 0

    for user_id in candidates:
        result = await client.search(user_id, probes[user_id], top_k)
        if not result.memories:
            chunks = written[user_id]
            raise ContractViolation(
                f"stored nothing: {chunks} chunk(s) were written for this context and "
                f"accepted, but the store returns no memories for a question about them. "
                f"An ingest that reports success against an empty store would be scored "
                f"as retrieval misses. Check the system's own logs for a write or "
                f"extraction step that failed while still answering 200."
            )
        log.info("liveness ok user=%s memories=%d", user_id, len(result.memories))

    return len(candidates)
