"""The one place the retry policy and the interface checks live (ARCH §4.3, §6).

Retry is keyed on the **exception type** the adapter raised, never on a status code.
An `AdapterError` is a retryable-class failure the suite's policy may retry; a
`ContractViolation` is a broken integration that fails the run loudly instead of
scoring as zero; anything else is an adapter bug and surfaces as itself, because
burying a `TypeError` under an exhausted retry policy hides the actual cause.

That split is what lets a system speak gRPC, or be a local library with no status
codes at all, and still get the same policy. Classifying HTTP is `adapters/http.py`'s
job and nobody else's.

Concurrency is bounded by a semaphore from the suite's `limits`, never a bare
`gather`, and `asyncio.CancelledError` is allowed to propagate.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from contract.adapter import AdapterError, ContractViolation, Memory, MemoryAdapter, Message
from orchestrator.runlog import get_logger
from orchestrator.suite import StageLimits

log = get_logger("adapter")


class RetryPolicy(BaseModel):
    """Per-stage retry, timeout, and concurrency bounds."""

    model_config = ConfigDict(frozen=True)

    attempts: int
    timeout_s: int
    workers: int
    backoff_s: float = 0.5

    @classmethod
    def for_stage(cls, stage: StageLimits) -> "RetryPolicy":
        """Build a policy from one stage's limits."""
        return cls(attempts=stage.attempts, timeout_s=stage.timeout_s, workers=stage.workers)


@dataclass(frozen=True)
class AddResult:
    """One completed `add`: what it cost, and when.

    `latency_ms` is measured on a monotonic clock, so it is immune to clock steps. The
    two timestamps are deliberately wall-clock instead: `verify` compares them across
    phases to prove ingest and retrieve never overlapped (D6), and a monotonic epoch is
    arbitrary and not comparable between processes.
    """

    attempts: int
    latency_ms: int
    started_at_ms: int
    ended_at_ms: int


@dataclass(frozen=True)
class SearchResult:
    """One completed `search`: the memories, verbatim and in rank order, plus its cost."""

    memories: Sequence[Memory]
    attempts: int
    latency_ms: int
    started_at_ms: int
    ended_at_ms: int


def check_memories(memories: Sequence[Memory], top_k: int) -> None:
    """Raise ContractViolation unless a search result is well-formed and within budget."""
    if len(memories) > top_k:
        raise ContractViolation(f"search returned {len(memories)} memories for top_k={top_k}")
    for memory in memories:
        if not isinstance(memory, Memory):
            raise ContractViolation(f"search returned a {type(memory).__name__}, expected a Memory")
        if not memory.content:
            # Empty content would score as a wrong answer, which reads as a memory
            # result rather than as the broken integration it is.
            raise ContractViolation("search returned a memory with empty content")


@dataclass(frozen=True)
class _Attempt:
    """Which attempt succeeded, and how long that attempt itself took."""

    attempts: int
    latency_ms: int


class AdapterClient:
    """Applies the retry policy and the interface checks to one adapter."""

    def __init__(self, adapter: MemoryAdapter, policy: RetryPolicy) -> None:
        self._adapter = adapter
        self._policy = policy
        self._gate = asyncio.Semaphore(policy.workers)

    async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> AddResult:
        """Store one chunk, retrying per policy; a ContractViolation fails the run."""
        # The wall-clock bounds span the whole time the call was outstanding, queue wait
        # included, because `verify` proves ingest and retrieve never overlap from them
        # (D6). `latency_ms` deliberately does not — see `_attempt`.
        started_wall = int(time.time() * 1000)
        outcome = await self._attempt(
            lambda: self._adapter.add(user_id, messages, chunk_id),
            what="add",
        )
        return AddResult(
            attempts=outcome.attempts,
            latency_ms=outcome.latency_ms,
            started_at_ms=started_wall,
            ended_at_ms=int(time.time() * 1000),
        )

    async def search(self, user_id: str, query: str, top_k: int) -> SearchResult:
        """Retrieve for one question, retrying per policy, and check the result."""
        started_wall = int(time.time() * 1000)
        memories: Sequence[Memory] = ()

        async def call() -> None:
            nonlocal memories
            memories = await self._adapter.search(user_id, query, top_k)

        outcome = await self._attempt(call, what="search")
        check_memories(memories, top_k)
        return SearchResult(
            memories=memories,
            attempts=outcome.attempts,
            latency_ms=outcome.latency_ms,
            started_at_ms=started_wall,
            ended_at_ms=int(time.time() * 1000),
        )

    async def close(self) -> None:
        """Close the adapter if it defines `close`; `close` is optional on the protocol."""
        closer = getattr(self._adapter, "close", None)
        if closer is not None:
            await closer()

    async def _attempt(self, call: Callable[[], Awaitable[None]], what: str) -> _Attempt:
        """Run `call` under the policy; return the winning attempt and its own duration.

        The duration is measured inside the gate, so it excludes time the call spent
        queued behind the worker bound. That wait is our scheduling: with more chunks
        than workers, charging it to the system would make the suite's own fan-out cap
        show up in the published latency percentiles.
        """
        last = "no attempt was made"

        for attempt in range(1, self._policy.attempts + 1):
            async with self._gate:
                started = time.monotonic()
                try:
                    async with asyncio.timeout(self._policy.timeout_s):
                        await call()
                except TimeoutError:
                    # A hung adapter is a retryable-class failure, not a broken
                    # integration: nothing about the contract has been violated yet.
                    last = f"{what} timed out after {self._policy.timeout_s}s"
                except AdapterError as err:
                    last = f"{what} failed: {err}"
                else:
                    return _Attempt(attempt, int((time.monotonic() - started) * 1000))

            # A silent retry is what makes a flaky system look healthy. Warn, so the
            # log shows the attempt that a successful record no longer mentions.
            log.warning("%s attempt %d/%d failed: %s", what, attempt, self._policy.attempts, last)
            if attempt < self._policy.attempts and self._policy.backoff_s:
                await asyncio.sleep(self._policy.backoff_s * attempt)

        raise AdapterError(f"{what} failed after {self._policy.attempts} attempts: {last}")


__all__ = [
    "AddResult",
    "AdapterClient",
    "AdapterError",
    "ContractViolation",
    "RetryPolicy",
    "SearchResult",
    "check_memories",
]
