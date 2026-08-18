"""The two privileged in-process baselines (D2, ARCH §10).

They need gold evidence and the whole corpus, which the contract deliberately does
not carry, so they read `plan.jsonl` and write `retrieve.jsonl` directly instead of
going through `ingest`/`retrieve`. Everything downstream is unchanged — that is what
keeps them comparable — and their rows are marked `in_process: true`.

This lives in `reference/`, never in a stage: no stage may know a system's name.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from orchestrator.artifacts import JsonlArtifact
from orchestrator.plan_records import ChunkRecord, QuestionRecord, parse_plan_record

Baseline = Literal["oracle_gold", "long_context"]


def _chunk_text(chunk: ChunkRecord) -> str:
    return "\n".join(str(m.get("content", "")) for m in chunk.messages)


def _memory(chunk: ChunkRecord, rank: int) -> dict[str, Any]:
    return {
        "content": _chunk_text(chunk),
        "score": round(1.0 / rank, 6),
        "source_ids": [chunk.chunk_id],
    }


def _gold_selector(question: QuestionRecord) -> Callable[[ChunkRecord], bool]:
    """A chunk is gold when `plan` resolved the question's evidence onto its id.

    The unit-to-chunk mapping is `orchestrator/evidence.py`'s job, done once at plan
    time; the ceiling just reads the result so it cannot drift from what recall scores.
    """
    wanted = set(question.evidence_chunks)
    return lambda chunk: chunk.chunk_id in wanted


def inject_retrieval(run_dir: Path, baseline: Baseline, top_k: int) -> int:
    """Write `retrieve.jsonl` directly for an in-process baseline; return rows written."""
    plan = JsonlArtifact(run_dir / "plan.jsonl")
    out = JsonlArtifact(run_dir / "retrieve.jsonl")
    done = out.read_ids()

    chunks: dict[str, list[ChunkRecord]] = {}
    questions: list[QuestionRecord] = []
    for raw in plan.stream():
        record = parse_plan_record(raw)
        if isinstance(record, ChunkRecord):
            chunks.setdefault(record.ctx_id, []).append(record)
        else:
            questions.append(record)

    written = 0
    for question in questions:
        if question.id in done:
            continue
        available = chunks.get(question.ctx_id, [])
        if baseline == "oracle_gold":
            is_gold = _gold_selector(question)
            selected = [c for c in available if is_gold(c)]
        else:
            selected = list(available)

        memories = [_memory(chunk, rank) for rank, chunk in enumerate(selected[:top_k], start=1)]
        out.append(
            {
                "id": question.id,
                "user_id": question.user_id,
                "query": question.question,
                "top_k": top_k,
                "attempts": 1,
                "latency_ms": 0,
                "memories": memories,
                # Which privileged baseline produced this row. The report marks it
                # `in_process: true`; this keeps the reason legible in the artifact too.
                "in_process": baseline,
            }
        )
        written += 1

    return written
