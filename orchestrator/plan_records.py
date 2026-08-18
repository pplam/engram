"""The shape of `plan.jsonl` (ARCH §5).

Two record kinds share one file, discriminated by `kind`: chunks to ingest and
questions to retrieve for. Both carry the `user_id` that isolates them.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from orchestrator.artifacts import Record


class _PlanRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    kind: str
    user_id: str
    ctx_id: str


class ChunkRecord(_PlanRecord):
    """One unit of ingestion."""

    kind: Literal["chunk"] = "chunk"
    chunk_id: str
    messages: list[dict[str, Any]]


class QuestionRecord(_PlanRecord):
    """One scorable question, its gold answer, and its evidence.

    `evidence_chunk_ids` holds the dataset's own handles in its own unit (message or
    session) so the annotation stays auditable; `evidence_chunks` holds those handles
    resolved to the chunk ids a system can echo in `x_source_ids`. Recall uses the
    latter — see `orchestrator/evidence.py`.
    """

    kind: Literal["question"] = "question"
    q_id: str
    question: str
    gold: list[str]
    options: list[str] | None = None
    evidence_chunk_ids: list[str] = []
    evidence_chunks: list[str] = []
    meta: dict[str, Any] = {}


PlanRecord = ChunkRecord | QuestionRecord


def parse_plan_record(raw: Record) -> PlanRecord:
    """Return the typed record for one `plan.jsonl` line."""
    kind = raw.get("kind")
    if kind == "chunk":
        return ChunkRecord.model_validate(raw)
    if kind == "question":
        return QuestionRecord.model_validate(raw)
    raise ValueError(f"unknown plan record kind {kind!r}")
