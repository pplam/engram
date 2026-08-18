"""Retry policy, bounded concurrency, and loud contract violations (§2.6).

Retry is keyed on the **exception type** an adapter raises, not on a status code. That
is the whole point of the interface: an adapter that talks gRPC, or a local library
with no status codes at all, still gets the suite's retry policy. Classifying a status
is `adapters/http.py`'s job and nobody else's.
"""

import asyncio
from collections.abc import Sequence

import pytest

from contract.adapter import AdapterError, ContractViolation, Memory, Message
from orchestrator.client import AdapterClient, RetryPolicy

POLICY = RetryPolicy(attempts=4, timeout_s=5, workers=2, backoff_s=0.0)
UID = "eval:r1:fixture:ctx0"
MESSAGES = [Message(role="user", content="the cat sat on the mat")]


class Scripted:
    """An adapter whose every call is dictated by the test."""

    def __init__(
        self,
        add_raises: list[Exception | None] | None = None,
        search_raises: list[Exception | None] | None = None,
        memories: Sequence[Memory] = (),
        delay_s: float = 0.0,
    ) -> None:
        self.add_calls = 0
        self.search_calls = 0
        self._add_raises = add_raises or []
        self._search_raises = search_raises or []
        self._memories = memories
        self._delay_s = delay_s

    async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> None:
        self.add_calls += 1
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        self._maybe_raise(self._add_raises, self.add_calls)

    async def search(self, user_id: str, query: str, top_k: int) -> Sequence[Memory]:
        self.search_calls += 1
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        self._maybe_raise(self._search_raises, self.search_calls)
        return self._memories

    @staticmethod
    def _maybe_raise(script: list[Exception | None], call: int) -> None:
        if call <= len(script) and script[call - 1] is not None:
            raise script[call - 1]  # type: ignore[misc]


def client(adapter: object, **kw: object) -> AdapterClient:
    policy = POLICY.model_copy(update=kw) if kw else POLICY
    return AdapterClient(adapter, policy=policy)  # type: ignore[arg-type]


async def test_add_reports_one_attempt_on_success() -> None:
    result = await client(Scripted()).add(UID, MESSAGES, chunk_id="c0")
    assert result.attempts == 1


async def test_add_records_latency() -> None:
    result = await client(Scripted()).add(UID, MESSAGES, chunk_id="c0")
    assert result.latency_ms >= 0


async def test_search_returns_the_memories_the_adapter_produced() -> None:
    found = [Memory(content="a cat named Mochi", score=0.9, source_ids=("c7",))]
    result = await client(Scripted(memories=found)).search(UID, "cat", top_k=10)
    assert list(result.memories) == found


async def test_search_preserves_rank_order() -> None:
    """Rank is position in the sequence, so reordering would change every metric."""
    found = [Memory(content="first"), Memory(content="second"), Memory(content="third")]
    result = await client(Scripted(memories=found)).search(UID, "x", top_k=10)
    assert [m.content for m in result.memories] == ["first", "second", "third"]


async def test_a_retryable_adapter_error_is_retried_then_succeeds() -> None:
    adapter = Scripted(add_raises=[AdapterError("upstream blip")])
    result = await client(adapter).add(UID, MESSAGES, chunk_id="c0")
    assert adapter.add_calls == 2
    assert result.attempts == 2


async def test_attempts_are_capped() -> None:
    adapter = Scripted(add_raises=[AdapterError("down")] * 10)
    with pytest.raises(AdapterError, match="4 attempts"):
        await client(adapter).add(UID, MESSAGES, chunk_id="c0")
    assert adapter.add_calls == 4


async def test_a_contract_violation_is_never_retried() -> None:
    """A broken integration is not a transient failure; retrying it just costs money."""
    adapter = Scripted(add_raises=[ContractViolation("no")] * 10)
    with pytest.raises(ContractViolation):
        await client(adapter).add(UID, MESSAGES, chunk_id="c0")
    assert adapter.add_calls == 1


async def test_an_unexpected_exception_type_is_not_retried() -> None:
    """An adapter bug must surface as itself, not as an exhausted retry policy."""
    adapter = Scripted(add_raises=[TypeError("adapter bug")] * 10)
    with pytest.raises(TypeError, match="adapter bug"):
        await client(adapter).add(UID, MESSAGES, chunk_id="c0")
    assert adapter.add_calls == 1


async def test_more_memories_than_top_k_is_a_contract_violation() -> None:
    adapter = Scripted(memories=[Memory(content=f"m{i}") for i in range(4)])
    with pytest.raises(ContractViolation, match="top_k"):
        await client(adapter).search(UID, "x", top_k=2)


async def test_a_memory_with_empty_content_is_a_contract_violation() -> None:
    """Empty content scores as a wrong answer, which would read as a memory result."""
    adapter = Scripted(memories=[Memory(content="")])
    with pytest.raises(ContractViolation, match="content"):
        await client(adapter).search(UID, "x", top_k=10)


async def test_a_non_memory_return_value_is_a_contract_violation() -> None:
    adapter = Scripted(memories=[{"content": "a dict, not a Memory"}])  # type: ignore[list-item]
    with pytest.raises(ContractViolation, match="Memory"):
        await client(adapter).search(UID, "x", top_k=10)


async def test_a_contract_violation_from_search_is_not_retried() -> None:
    adapter = Scripted(search_raises=[ContractViolation("leak")] * 10)
    with pytest.raises(ContractViolation):
        await client(adapter).search(UID, "x", top_k=10)
    assert adapter.search_calls == 1


async def test_a_timeout_bounds_a_hanging_adapter() -> None:
    """An adapter that never returns must not hang the run forever."""
    adapter = Scripted(delay_s=10)
    with pytest.raises(AdapterError, match="timed out"):
        await client(adapter, attempts=1, timeout_s=0).search(UID, "x", top_k=10)


async def test_a_timeout_is_retried_then_reported() -> None:
    adapter = Scripted(delay_s=10)
    with pytest.raises(AdapterError, match="2 attempts"):
        await client(adapter, attempts=2, timeout_s=0).search(UID, "x", top_k=10)
    assert adapter.search_calls == 2


async def test_concurrency_is_bounded_by_workers() -> None:
    in_flight = 0
    peak = 0

    class Counting:
        async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> None:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1

        async def search(self, user_id: str, query: str, top_k: int) -> Sequence[Memory]:
            return ()

    bounded = client(Counting(), workers=2)
    await asyncio.gather(*(bounded.add(UID, MESSAGES, chunk_id=f"c{i}") for i in range(6)))
    assert peak <= 2


async def test_cancellation_propagates() -> None:
    """`asyncio.CancelledError` must bubble so `bench stop` drains rather than hangs."""
    bounded = client(Scripted(delay_s=10), timeout_s=30)
    task = asyncio.create_task(bounded.add(UID, MESSAGES, chunk_id="c0"))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_call_records_when_it_started_and_ended() -> None:
    """`verify` needs wall-clock bounds to prove ingest and retrieve never overlap (D6)."""
    result = await client(Scripted()).add(UID, MESSAGES, chunk_id="c0")
    assert result.started_at_ms <= result.ended_at_ms


async def test_call_timestamps_are_wall_clock_not_monotonic() -> None:
    """A monotonic clock has an arbitrary epoch and cannot be compared across phases."""
    import time

    before = int(time.time() * 1000)
    result = await client(Scripted()).add(UID, MESSAGES, chunk_id="c0")
    assert result.started_at_ms >= before


async def test_close_closes_an_adapter_that_defines_it() -> None:
    class Closable:
        closed = False

        async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> None:
            pass

        async def search(self, user_id: str, query: str, top_k: int) -> Sequence[Memory]:
            return ()

        async def close(self) -> None:
            self.closed = True

    adapter = Closable()
    await client(adapter).close()
    assert adapter.closed


async def test_close_is_a_no_op_for_an_adapter_without_it() -> None:
    """`close` is optional on the protocol, so calling it must never be an error."""
    await client(Scripted()).close()


async def test_latency_excludes_time_spent_waiting_for_a_worker() -> None:
    """Queue wait is our scheduling, not the system's speed.

    `latency_ms` becomes the published ingest/retrieve percentiles. With more chunks
    than workers the queued ones wait, and counting that wait attributes the harness's
    own worker cap to the system under test — p95 then measures our fan-out bound.
    """
    adapter = Scripted(delay_s=0.20)
    one = client(adapter, workers=1)
    results = await asyncio.gather(
        one.add(UID, MESSAGES, chunk_id="c0"),
        one.add(UID, MESSAGES, chunk_id="c1"),
    )
    # The second call waited ~200ms for the worker, then took ~200ms itself. Its latency
    # must describe the call, so both readings land near the delay rather than one near
    # double it.
    assert max(r.latency_ms for r in results) < 350


async def test_timestamps_still_span_the_wait_so_phase_ordering_holds() -> None:
    """`verify` proves ingest and retrieve never overlap from these bounds (D6).

    They must therefore cover the whole time the call was outstanding, queue wait
    included — the opposite of what `latency_ms` measures.
    """
    adapter = Scripted(delay_s=0.10)
    one = client(adapter, workers=1)
    results = await asyncio.gather(
        one.add(UID, MESSAGES, chunk_id="c0"),
        one.add(UID, MESSAGES, chunk_id="c1"),
    )
    span = max(r.ended_at_ms for r in results) - min(r.started_at_ms for r in results)
    assert span >= 200
