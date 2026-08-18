"""Reading `retrieve.jsonl` across the record-shape change (Phase 6, step 5).

`data`/`x_source_ids` became `memories`/`source_ids`. Both shapes must read, because
`bench rescore` over a bundle produced before the refactor is Phase 6's acceptance
criterion — a published row has to stay recomputable from the artifacts that produced it.
"""

from orchestrator.retrieval_records import memories_of, source_ids_of

NEW = {
    "id": "q1",
    "memories": [{"content": "a cat named Mochi", "score": 0.9, "source_ids": ["c7"]}],
}
OLD = {
    "id": "q1",
    "data": [{"id": "m1", "content": "a cat named Mochi", "score": 0.9, "x_source_ids": ["c7"]}],
}


def test_reads_the_current_memories_key() -> None:
    assert [m["content"] for m in memories_of(NEW)] == ["a cat named Mochi"]


def test_reads_a_pre_refactor_data_key() -> None:
    assert [m["content"] for m in memories_of(OLD)] == ["a cat named Mochi"]


def test_a_row_with_neither_key_reads_as_no_memories() -> None:
    assert memories_of({"id": "q1"}) == []


def test_an_empty_retrieval_reads_as_no_memories() -> None:
    """`no_memory` produces these for every question; they are a measurement, not a gap."""
    assert memories_of({"id": "q1", "memories": []}) == []


def test_the_current_key_wins_when_both_are_present() -> None:
    """A transitional bundle must not be scored twice or from the stale half."""
    row = {"memories": [{"content": "new"}], "data": [{"content": "old"}]}
    assert [m["content"] for m in memories_of(row)] == ["new"]


def test_source_ids_reads_the_current_key() -> None:
    assert source_ids_of({"content": "x", "source_ids": ["c1", "c2"]}) == ["c1", "c2"]


def test_source_ids_reads_the_pre_refactor_key() -> None:
    assert source_ids_of({"content": "x", "x_source_ids": ["c1"]}) == ["c1"]


def test_a_memory_with_no_source_ids_reads_as_empty() -> None:
    assert source_ids_of({"content": "x"}) == []


def test_a_null_source_ids_reads_as_empty() -> None:
    """The old mock wrote `null` rather than omitting the key."""
    assert source_ids_of({"content": "x", "x_source_ids": None}) == []


def test_source_ids_are_stringified() -> None:
    assert source_ids_of({"content": "x", "source_ids": [7]}) == ["7"]
