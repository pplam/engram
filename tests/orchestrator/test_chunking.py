"""Chunking is pure and deterministic; message limit wins over word limit (D4, §2.2)."""

import pytest

from orchestrator.chunking import Chunk, chunk_messages
from orchestrator.suite import Chunking

MSG = {"role": "user", "content": "one two three", "timestamp": 1}


def cfg(messages: int = 20, words: int = 2000) -> Chunking:
    return Chunking(
        max_messages_per_chunk=messages,
        max_words_per_chunk=words,
        boundary="message_or_sentence",
    )


def msgs(n: int, content: str = "one two three") -> list[dict[str, object]]:
    return [{"role": "user", "content": content, "timestamp": i} for i in range(n)]


def test_empty_input_produces_no_chunks() -> None:
    assert chunk_messages([], cfg()) == []


def test_fewer_messages_than_the_limit_produce_one_chunk() -> None:
    chunks = chunk_messages(msgs(3), cfg(messages=20))
    assert len(chunks) == 1
    assert len(chunks[0].messages) == 3


def test_exactly_at_the_message_limit_produces_one_chunk() -> None:
    chunks = chunk_messages(msgs(20), cfg(messages=20))
    assert len(chunks) == 1
    assert len(chunks[0].messages) == 20


def test_one_over_the_message_limit_produces_two_chunks() -> None:
    chunks = chunk_messages(msgs(21), cfg(messages=20))
    assert [len(c.messages) for c in chunks] == [20, 1]


def test_word_limit_cuts_inside_the_message_window() -> None:
    """Three words per message, 6-word limit -> 2 messages per chunk despite messages=20."""
    chunks = chunk_messages(msgs(4), cfg(messages=20, words=6))
    assert [len(c.messages) for c in chunks] == [2, 2]


def test_message_limit_wins_when_both_bounds_are_reached() -> None:
    """messages=2 and words=6 both permit 2 messages; the message bound is primary."""
    chunks = chunk_messages(msgs(4), cfg(messages=2, words=6))
    assert [len(c.messages) for c in chunks] == [2, 2]


def test_message_limit_wins_over_a_more_generous_word_limit() -> None:
    chunks = chunk_messages(msgs(6), cfg(messages=2, words=10_000))
    assert [len(c.messages) for c in chunks] == [2, 2, 2]


def test_chunk_ids_are_sequential_and_prefixed() -> None:
    chunks = chunk_messages(msgs(5), cfg(messages=2))
    assert [c.chunk_id for c in chunks] == ["c0", "c1", "c2"]


def test_a_message_longer_than_the_word_limit_splits_at_a_sentence_boundary() -> None:
    long = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
    chunks = chunk_messages([{"role": "user", "content": long, "timestamp": 1}], cfg(words=4))
    contents = [m["content"] for c in chunks for m in c.messages]
    assert all(part.endswith(".") for part in contents)
    assert " ".join(contents) == long


def test_split_parts_keep_the_parent_chunk_id_with_an_ordinal_suffix() -> None:
    long = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
    chunks = chunk_messages([{"role": "user", "content": long, "timestamp": 1}], cfg(words=4))
    assert [c.chunk_id for c in chunks] == ["c0.0", "c0.1", "c0.2"]


def test_split_preserves_the_parent_role_and_timestamp() -> None:
    long = "Alpha beta gamma. Delta epsilon zeta."
    chunks = chunk_messages([{"role": "assistant", "content": long, "timestamp": 42}], cfg(words=4))
    for chunk in chunks:
        assert chunk.messages[0]["role"] == "assistant"
        assert chunk.messages[0]["timestamp"] == 42


def test_a_single_sentence_over_the_word_limit_is_not_split_further() -> None:
    """No sentence boundary exists, so the oversized sentence is emitted whole."""
    long = "alpha beta gamma delta epsilon zeta"
    chunks = chunk_messages([{"role": "user", "content": long, "timestamp": 1}], cfg(words=2))
    assert len(chunks) == 1
    assert chunks[0].messages[0]["content"] == long


def test_split_message_is_not_packed_with_other_messages() -> None:
    """A split parent occupies its own chunks so provenance stays unambiguous."""
    long = "Alpha beta gamma. Delta epsilon zeta."
    messages = [MSG, {"role": "user", "content": long, "timestamp": 2}, MSG]
    chunks = chunk_messages(messages, cfg(messages=20, words=4))
    ids = [c.chunk_id for c in chunks]
    assert ids == ["c0", "c1.0", "c1.1", "c2"]


def test_chunking_is_deterministic() -> None:
    messages = msgs(47)
    assert chunk_messages(messages, cfg(messages=7)) == chunk_messages(messages, cfg(messages=7))


def test_every_message_survives_chunking() -> None:
    messages = msgs(47)
    packed = [m for c in chunk_messages(messages, cfg(messages=7)) for m in c.messages]
    assert packed == messages


def test_chunk_is_frozen() -> None:
    chunk = chunk_messages(msgs(1), cfg())[0]
    with pytest.raises((AttributeError, TypeError)):
        chunk.chunk_id = "other"  # type: ignore[misc]


def test_returns_chunk_instances() -> None:
    assert all(isinstance(c, Chunk) for c in chunk_messages(msgs(3), cfg(messages=2)))
