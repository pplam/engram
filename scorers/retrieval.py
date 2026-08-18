"""Retrieval-quality metrics on the exact track (ARCH §7.2).

Exact means the system reported `source_ids`, so gold evidence is compared by ID.
The fuzzy content-matching track is ARCH §15.2 and unresolved — it is not
implemented here rather than implemented and quietly published.

Every figure is labelled with the track that produced it, and the *unit* of the
gold IDs (message vs session) is a dataset property carried by the report, not by
this function.
"""

import math
from dataclasses import dataclass
from typing import Any, Literal

from orchestrator.retrieval_records import source_ids_of

Track = Literal["exact", "unavailable"]


@dataclass(frozen=True)
class RetrievalOutcome:
    """Retrieval metrics for one question. `None` means not measurable, not zero."""

    track: Track
    recall_at_k: float | None
    precision_at_k: float | None
    mrr: float | None
    ndcg_at_k: float | None
    content_tokens: int
    retrieved: int


def supports_exact_track(memories: list[dict[str, Any]]) -> bool:
    """True when at least one returned memory reports `source_ids`."""
    return any(source_ids_of(item) for item in memories)


def _ndcg(hits: list[bool], gold_count: int) -> float:
    gain = sum(1 / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1) if hit)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(gold_count, len(hits)) + 1))
    return gain / ideal if ideal else 0.0


def score_retrieval(
    memories: list[dict[str, Any]], gold: tuple[str, ...], k: int
) -> RetrievalOutcome:
    """Return retrieval metrics for one question's verbatim memories, in rank order."""
    considered = memories[:k]
    content_tokens = sum(len(str(item.get("content", "")).split()) for item in considered)

    if not gold or (considered and not supports_exact_track(considered)):
        track: Track = "exact" if gold and supports_exact_track(considered) else "unavailable"
        return RetrievalOutcome(
            track=track,
            recall_at_k=None,
            precision_at_k=None,
            mrr=None,
            ndcg_at_k=None,
            content_tokens=content_tokens,
            retrieved=len(considered),
        )

    expected = set(gold)
    hits = [bool(expected & set(source_ids_of(item))) for item in considered]
    found = {sid for item in considered for sid in source_ids_of(item)} & expected

    first = next((rank for rank, hit in enumerate(hits, start=1) if hit), 0)
    return RetrievalOutcome(
        track="exact",
        recall_at_k=len(found) / len(expected),
        precision_at_k=(sum(hits) / len(considered)) if considered else 0.0,
        mrr=1 / first if first else 0.0,
        ndcg_at_k=_ndcg(hits, len(expected)),
        content_tokens=content_tokens,
        retrieved=len(considered),
    )
