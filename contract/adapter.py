"""The adapter interface — the entire integration surface (ARCH §4.1-4.3).

An adapter is a **client, not a server**. It implements two methods against whatever
SDK its system provides; the harness imports it and calls it directly. There is no
wire format, no envelope, and nothing to run.

Two failure kinds, never conflated:

- `AdapterError` — a transport blip or an upstream 5xx. The suite's retry policy may
  retry it, and only an exhausted policy fails the run.
- `ContractViolation` — a broken integration: more than `top_k` memories, empty
  content, another user's content. This fails the run **loudly** rather than scoring
  as zero, because a broken integration is not a memory result.

`Message` and `Memory` are frozen dataclasses rather than Pydantic models on purpose.
They are constructed once per message and once per memory in the hot path, and there
is no longer any untrusted wire input to validate — validation belongs at the artifact
boundary, which `orchestrator/artifacts.py` owns.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class AdapterError(Exception):
    """A retryable failure reaching the system: transport, timeout, or upstream 5xx."""


class ContractViolation(Exception):
    """The system broke the interface. Fails the run loudly, never scores as zero."""


@dataclass(frozen=True)
class Message:
    """One conversation turn handed to `add`."""

    role: str
    content: str
    timestamp: int | None = None


@dataclass(frozen=True)
class Memory:
    """One memory returned by `search`.

    `score` is whatever the system reports and is never interpreted — rank order is
    the position in the returned sequence. `source_ids` carries the `chunk_id`s the
    memory derives from; without them recall reads `unavailable` rather than zero.
    """

    content: str
    score: float | None = None
    source_ids: Sequence[str] = field(default_factory=tuple)


@runtime_checkable
class MemoryAdapter(Protocol):
    """What a memory system must implement to be benchmarked.

    Three members are deliberately absent, because requiring any of them would add a
    no-op to every implementation. The harness calls each one when an adapter happens to
    define it:

    - `async def close() -> None` — release whatever the adapter holds.
    - `stores_nothing: bool` — declares that an empty store is this system's measurement
      rather than a dropped write, which exempts it from the post-ingest gate (ARCH §8).
    - `async def use_model_key(value: str) -> None` — present `value` on the system's own
      model calls, so the gateway can attribute what it spent (D6(a)). A system that
      cannot take a credential at runtime simply omits it and its cost reads
      `unavailable`, which is honest; a zero would not be.
    """

    async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> None:
        """Store `messages` under `user_id`; return only once they are searchable."""
        ...

    async def search(self, user_id: str, query: str, top_k: int) -> Sequence[Memory]:
        """Return at most `top_k` memories for `user_id`, best first. Never an answer."""
        ...


__all__ = [
    "AdapterError",
    "ContractViolation",
    "Memory",
    "MemoryAdapter",
    "Message",
]
