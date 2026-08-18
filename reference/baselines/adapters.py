"""The two baselines that are ordinary adapters (ARCH §10).

`no_memory` and `bm25` implement `MemoryAdapter` exactly as a third party's system
would, which is the cheapest possible check that the interface is implementable. They
need no registry entry and no socket.

`oracle_gold` and `long_context` cannot be adapters: they need gold evidence and the
whole corpus, which the interface deliberately does not carry. They stay in
`in_process.py` and emit `retrieve.jsonl` directly.
"""

from collections.abc import Callable, Sequence

from contract.adapter import Memory, Message

from .bm25 import Bm25System


class NoMemoryAdapter:
    """Accepts every write and forgets it. The floor for every dataset.

    Both methods ignore most of their arguments on purpose — that is the baseline.
    """

    # Exempts this baseline from the post-ingest gate, which otherwise reads an empty
    # store as an integration that accepted writes and dropped them (ARCH §8). Declared
    # here rather than decided by name in `orchestrator/`, which carries no per-system
    # branching: storing nothing is this baseline's measurement, and only the system
    # itself can say so. Optional on the protocol, like `close`.
    stores_nothing = True

    async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> None:
        """Discard the write; returning cleanly is honest because nothing was stored."""

    async def search(self, user_id: str, query: str, top_k: int) -> Sequence[Memory]:
        """Return no memories, always."""
        return ()


class Bm25Adapter:
    """Okapi BM25 over per-user chunk documents. Answers "does this beat lexical search?"."""

    def __init__(self) -> None:
        self._index = Bm25System()

    async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> None:
        """Index one document per message under `user_id`."""
        self._index.add(user_id, [m.content for m in messages], chunk_id=chunk_id)

    async def search(self, user_id: str, query: str, top_k: int) -> Sequence[Memory]:
        """Return the top_k best-scoring documents for `user_id`, best first."""
        return [
            Memory(
                content=hit.content,
                score=hit.score,
                source_ids=(hit.chunk_id,) if hit.chunk_id else (),
            )
            for hit in self._index.search(user_id, query, top_k)
        ]


BaselineAdapter = NoMemoryAdapter | Bm25Adapter

# The in-repo baselines that implement the interface. Reachable without a registry entry
# or a socket, while going through the identical code path a registered system does — a
# baseline that took a shortcut would not be measuring the same thing.
BASELINE_ADAPTERS: dict[str, Callable[[], BaselineAdapter]] = {
    "no_memory": NoMemoryAdapter,
    "bm25": Bm25Adapter,
}


def build_baseline(name: str) -> BaselineAdapter:
    """Return the in-repo baseline adapter called `name`."""
    if name not in BASELINE_ADAPTERS:
        raise KeyError(
            f"{name!r} is not an in-repo baseline adapter; expected one of "
            f"{sorted(BASELINE_ADAPTERS)}"
        )
    return BASELINE_ADAPTERS[name]()


__all__ = [
    "BASELINE_ADAPTERS",
    "BaselineAdapter",
    "Bm25Adapter",
    "NoMemoryAdapter",
    "build_baseline",
]
