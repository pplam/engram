"""Gold evidence is annotated in dataset units and must map onto chunk ids (ARCH §7.2).

LoCoMo names messages (`m0`), LongMemEval names sessions (`sess0`); ingestion works
in chunks (`c0`). Recall compares against `x_source_ids`, which carry chunk ids, so
the plan records the expansion. A gold unit spanning several chunks expands to all
of them.
"""

from orchestrator.chunking import Chunk
from orchestrator.evidence import expand_evidence


def chunk(chunk_id: str, sources: tuple[int, ...], *messages: dict[str, str]) -> Chunk:
    return Chunk(chunk_id=chunk_id, messages=tuple(messages), source_indices=sources)


def test_a_message_index_maps_to_the_chunk_holding_it() -> None:
    chunks = [
        chunk("c0", (0, 1), {"content": "a"}, {"content": "b"}),
        chunk("c1", (2,), {"content": "c"}),
    ]
    assert expand_evidence(("m2",), chunks, "message") == ("c1",)


def test_message_indices_are_positions_in_the_context_not_in_the_chunk() -> None:
    chunks = [
        chunk("c0", (0, 1), {"content": "a"}, {"content": "b"}),
        chunk("c1", (2,), {"content": "c"}),
    ]
    assert expand_evidence(("m0", "m1"), chunks, "message") == ("c0",)


def test_a_session_expands_to_every_chunk_derived_from_it() -> None:
    """Coarser evidence is a systematically easier target; it must not be lossy."""
    chunks = [
        chunk("c0", (0,), {"content": "a", "session_id": "sess0"}),
        chunk("c1", (1,), {"content": "b", "session_id": "sess0"}),
        chunk("c2", (2,), {"content": "c", "session_id": "sess1"}),
    ]
    assert expand_evidence(("sess0",), chunks, "session") == ("c0", "c1")


def test_a_chunk_spanning_two_sessions_counts_for_both() -> None:
    chunks = [
        chunk(
            "c0",
            (0, 1),
            {"content": "a", "session_id": "sess0"},
            {"content": "b", "session_id": "sess1"},
        )
    ]
    assert expand_evidence(("sess1",), chunks, "session") == ("c0",)


def test_an_unknown_unit_expands_to_nothing() -> None:
    """A dataset annotation naming no ingested content must not be silently correct."""
    assert expand_evidence(("m9",), [chunk("c0", (0,), {"content": "a"})], "message") == ()


def test_no_evidence_expands_to_nothing() -> None:
    assert expand_evidence((), [chunk("c0", (0,), {"content": "a"})], "message") == ()


def test_an_oversized_message_keeps_provenance_through_the_split() -> None:
    """D4 splits one message into c1.0/c1.1; both still answer for that message."""
    chunks = [
        chunk("c0", (0,), {"content": "a"}),
        chunk("c1.0", (1,), {"content": "b"}),
        chunk("c1.1", (1,), {"content": "c"}),
    ]
    assert expand_evidence(("m1",), chunks, "message") == ("c1.0", "c1.1")


def test_a_malformed_message_handle_expands_to_nothing() -> None:
    assert expand_evidence(("mx",), [chunk("c0", (0,), {"content": "a"})], "message") == ()


def test_a_message_handle_may_be_the_datasets_own_id() -> None:
    """LoCoMo names messages `D1:3`, not by position, and recall is compared by ID.

    Without this, every LoCoMo question expands to no chunks and silently drops to the
    `unavailable` track — a whole dataset reporting no recall for a fixable reason.
    """
    chunks = [
        Chunk(
            chunk_id="c0",
            messages=({"dia_id": "D1:1"}, {"dia_id": "D1:2"}),
            source_indices=(0, 1),
        ),
        Chunk(chunk_id="c1", messages=({"dia_id": "D1:3"},), source_indices=(2,)),
    ]
    assert expand_evidence(("D1:3",), chunks, "message") == ("c1",)


def test_positional_message_handles_still_work() -> None:
    """The fixtures annotate by position, so both spellings must resolve."""
    chunks = [
        Chunk(chunk_id="c0", messages=({"dia_id": "D1:1"},), source_indices=(0,)),
        Chunk(chunk_id="c1", messages=({"dia_id": "D1:2"},), source_indices=(1,)),
    ]
    assert expand_evidence(("m1",), chunks, "message") == ("c1",)
