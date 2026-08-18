"""The adapter interface is the whole integration surface (§4.1-4.3).

These records cross the boundary between the harness and a foreign SDK, so they are
frozen dataclasses rather than Pydantic models: they are constructed once per message
and once per memory in the hot path, and there is no untrusted wire input left to
validate. Validation moved to the artifact boundary, which `orchestrator/artifacts.py`
owns.
"""

import dataclasses
from collections.abc import Sequence

import pytest

from contract.adapter import AdapterError, ContractViolation, Memory, MemoryAdapter, Message


class Conformant:
    """The minimum that satisfies the protocol."""

    async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> None:
        """Store nothing; the signature is the point."""

    async def search(self, user_id: str, query: str, top_k: int) -> Sequence[Memory]:
        """Return nothing; the signature is the point."""
        return ()


class MissingSearch:
    """Half an adapter: `add` only."""

    async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> None:
        """Store nothing."""


def test_a_memory_needs_only_content() -> None:
    """Score and source_ids are optional: a system may support neither."""
    memory = Memory(content="the cat sat on the mat")
    assert memory.score is None
    assert memory.source_ids == ()


def test_a_message_needs_only_role_and_content() -> None:
    assert Message(role="user", content="hello").timestamp is None


@pytest.mark.parametrize(
    ("record", "field", "value"),
    [
        (Memory(content="x"), "content", "y"),
        (Message(role="user", content="x"), "content", "y"),
    ],
)
def test_records_are_frozen(record: object, field: str, value: str) -> None:
    """An adapter must not be able to mutate a record the harness already recorded."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(record, field, value)


def test_a_class_with_both_methods_satisfies_the_protocol() -> None:
    assert isinstance(Conformant(), MemoryAdapter)


def test_a_class_missing_search_does_not_satisfy_the_protocol() -> None:
    assert not isinstance(MissingSearch(), MemoryAdapter)


def test_close_is_optional() -> None:
    """Most adapters hold nothing to close, so the protocol must not demand it."""
    assert isinstance(Conformant(), MemoryAdapter)
    assert not hasattr(Conformant(), "close")


def test_the_two_failure_kinds_are_distinct() -> None:
    """A retryable upstream blip and a broken integration must never be conflated (§4.3)."""
    assert not issubclass(AdapterError, ContractViolation)
    assert not issubclass(ContractViolation, AdapterError)
