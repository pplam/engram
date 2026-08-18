"""Retrieval metrics on the exact (`x_source_ids`) track (§2.10, §7.2).

The fuzzy matcher is ARCH §15.2 and unresolved, so it is gated behind a flag and
its numbers are labelled unpublishable.
"""

import pytest

from scorers.retrieval import (
    RetrievalOutcome,
    score_retrieval,
    supports_exact_track,
)


def memories(*chunk_ids: str | None) -> list[dict[str, object]]:
    return [
        {
            "id": f"mem_{i}",
            "content": f"content {i}",
            "x_source_ids": [c] if c else None,
        }
        for i, c in enumerate(chunk_ids)
    ]


def test_perfect_retrieval_scores_recall_one() -> None:
    outcome = score_retrieval(memories("c1", "c2"), gold=("c1", "c2"), k=10)
    assert outcome.recall_at_k == 1.0


def test_half_the_evidence_scores_recall_one_half() -> None:
    outcome = score_retrieval(memories("c1", "c9"), gold=("c1", "c2"), k=10)
    assert outcome.recall_at_k == 0.5


def test_no_evidence_scores_recall_zero() -> None:
    outcome = score_retrieval(memories("c8", "c9"), gold=("c1", "c2"), k=10)
    assert outcome.recall_at_k == 0.0


def test_precision_counts_gold_items_among_those_returned() -> None:
    outcome = score_retrieval(memories("c1", "c9", "c8", "c7"), gold=("c1",), k=10)
    assert outcome.precision_at_k == 0.25


def test_mrr_is_the_reciprocal_rank_of_the_first_gold_hit() -> None:
    outcome = score_retrieval(memories("c9", "c1"), gold=("c1",), k=10)
    assert outcome.mrr == 0.5


def test_mrr_is_one_when_the_first_item_is_gold() -> None:
    assert score_retrieval(memories("c1", "c9"), gold=("c1",), k=10).mrr == 1.0


def test_mrr_is_zero_when_nothing_is_gold() -> None:
    assert score_retrieval(memories("c8"), gold=("c1",), k=10).mrr == 0.0


def test_ndcg_rewards_ranking_gold_higher() -> None:
    high = score_retrieval(memories("c1", "c9"), gold=("c1",), k=10).ndcg_at_k
    low = score_retrieval(memories("c9", "c1"), gold=("c1",), k=10).ndcg_at_k
    assert high is not None and low is not None
    assert high > low


def test_ndcg_is_one_for_ideal_ranking() -> None:
    assert score_retrieval(memories("c1", "c2"), gold=("c1", "c2"), k=10).ndcg_at_k == 1.0


def test_only_the_first_k_items_are_considered() -> None:
    outcome = score_retrieval(memories("c9", "c9", "c1"), gold=("c1",), k=2)
    assert outcome.recall_at_k == 0.0


def test_retrieved_content_tokens_are_counted_for_context_efficiency() -> None:
    outcome = score_retrieval(memories("c1"), gold=("c1",), k=10)
    assert outcome.content_tokens > 0


def test_a_question_without_gold_evidence_reports_none_not_zero() -> None:
    """A dataset without evidence must not be scored as a retrieval failure."""
    outcome = score_retrieval(memories("c1"), gold=(), k=10)
    assert outcome.recall_at_k is None
    assert outcome.precision_at_k is None
    assert outcome.mrr is None


def test_empty_retrieval_scores_zero_when_gold_exists() -> None:
    outcome = score_retrieval([], gold=("c1",), k=10)
    assert outcome.recall_at_k == 0.0
    assert outcome.content_tokens == 0


def test_multiple_source_ids_on_one_memory_all_count() -> None:
    data = [{"id": "m1", "content": "x", "x_source_ids": ["c1", "c2"]}]
    assert score_retrieval(data, gold=("c1", "c2"), k=10).recall_at_k == 1.0


def test_a_system_without_source_ids_is_not_on_the_exact_track() -> None:
    assert supports_exact_track(memories(None, None)) is False


def test_a_system_echoing_source_ids_is_on_the_exact_track() -> None:
    assert supports_exact_track(memories("c1")) is True


def test_track_is_reported_so_readers_know_which_number_they_have() -> None:
    assert score_retrieval(memories("c1"), gold=("c1",), k=10).track == "exact"


def test_a_system_without_source_ids_reports_recall_as_unavailable() -> None:
    outcome = score_retrieval(memories(None), gold=("c1",), k=10)
    assert outcome.track == "unavailable"
    assert outcome.recall_at_k is None


def test_outcome_is_frozen() -> None:
    outcome = score_retrieval(memories("c1"), gold=("c1",), k=10)
    with pytest.raises((AttributeError, TypeError)):
        outcome.recall_at_k = 0.0  # type: ignore[misc]


def test_returns_a_retrieval_outcome() -> None:
    assert isinstance(score_retrieval(memories("c1"), gold=("c1",), k=10), RetrievalOutcome)


def test_scoring_is_deterministic() -> None:
    data = memories("c1", "c9")
    assert score_retrieval(data, gold=("c1",), k=10) == score_retrieval(data, gold=("c1",), k=10)
