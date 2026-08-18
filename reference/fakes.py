"""In-repo fake adapters (ARCH §10, §1.4).

`FakeMemory()` with no flags is a minimal conformant adapter: a dict keyed by
`user_id`, substring scoring, honours `top_k`, echoes `source_ids`. It exists so the
conformance suite has something to pass against in CI without a socket.

Every other flag makes it misbehave in exactly one way. Those are how we test that
conformance actually *fails* — a check that never fires is not a check. One flag each,
deliberately, so a check firing on the wrong fake means the check is wrong.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from contract.adapter import AdapterError, Memory, Message

_OVERSHOOT_BY = 5


class TransientFailure(AdapterError):
    """A retryable failure the fake raises on demand, to exercise the retry policy."""


@dataclass
class _Doc:
    content: str
    chunk_id: str
    visible: bool = True


@dataclass
class FakeMemory:
    """A conformant in-memory adapter, plus one flag per way of breaking the contract.

    Scoring is deliberately trivial — how many query words appear in the document. It
    only has to rank plausibly enough that "best first" is a meaningful claim; a real
    ranking function belongs in `bm25`, which is a baseline rather than a fixture.
    """

    leak_across_users: bool = False
    overshoot_top_k: bool = False
    empty_content: bool = False
    drop_source_ids: bool = False
    index_late: bool = False
    fail_adds: int = 0
    # Not a flaw, unlike the flags above: `extractive` is a second conformant shape.
    # mem0 stores LLM-derived facts, so it keeps only what reads as a fact and rewrites
    # what it keeps. Modelled here so the conformance suite is held to a probe both
    # storage styles can satisfy, rather than to verbatim recall.
    extractive: bool = False
    # A system that accepts every `add` and stores none of it. This is what an extractive
    # system looks like when the probe is not fact-shaped, and it exists to keep the
    # isolation check from concluding anything from an empty store.
    store_nothing: bool = False

    _docs: dict[str, list[_Doc]] = field(default_factory=dict)
    _failures: int = 0

    async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> None:
        """Store one document per message under `user_id`."""
        if self._failures < self.fail_adds:
            self._failures += 1
            raise TransientFailure(f"transient failure {self._failures} of {self.fail_adds}")

        if self.store_nothing:
            return

        for message in messages:
            content = self._extract(message.content) if self.extractive else message.content
            if content is None:
                continue
            self._docs.setdefault(user_id, []).append(
                _Doc(
                    content=content,
                    chunk_id=chunk_id,
                    # `index_late` returns from `add` before the content is searchable,
                    # which is the whole `late_indexing` failure mode.
                    visible=not self.index_late,
                )
            )

    def _extract(self, content: str) -> str | None:
        """Return `content` as a paraphrased fact, or None if it reads as no fact at all.

        A stand-in for an LLM: keep anything asserting something about the user, drop the
        rest, and restate what is kept in the third person so verbatim matching fails
        while the distinguishing tokens survive. Both halves matter — a fake that only
        dropped content would let a check pass by matching the input string.
        """
        lowered = content.lower()
        if not any(marker in f" {lowered} " for marker in (" is ", " are ", " my ", " i ")):
            return None
        return "User reports: " + content.replace("My ", "their ").replace("my ", "their ")

    async def search(self, user_id: str, query: str, top_k: int) -> Sequence[Memory]:
        """Return the best-matching visible documents for `user_id`, best first."""
        pool = self._pool(user_id)
        wanted = Counter(query.lower().split())

        scored: list[tuple[float, int, _Doc]] = []
        for position, doc in enumerate(pool):
            if not doc.visible:
                # A late-indexing system reveals content on a later read, not this one.
                doc.visible = True
                continue
            words = Counter(doc.content.lower().split())
            score = float(sum(count for term, count in words.items() if term in wanted))
            if score > 0:
                scored.append((score, position, doc))

        # Position breaks ties so the order is total: an unstable sort would make
        # rank-stability metrics measure our fixture rather than the system.
        scored.sort(key=lambda item: (-item[0], item[1]))

        budget = top_k + _OVERSHOOT_BY if self.overshoot_top_k else top_k
        return [self._memory(score, doc) for score, _, doc in scored[:budget]]

    def _pool(self, user_id: str) -> list[_Doc]:
        if self.leak_across_users:
            # Ignoring the isolation boundary entirely — the violation `verify`
            # invalidates a whole run over.
            return [doc for docs in self._docs.values() for doc in docs]
        return self._docs.get(user_id, [])

    def _memory(self, score: float, doc: _Doc) -> Memory:
        return Memory(
            content="" if self.empty_content else doc.content,
            score=score,
            source_ids=() if self.drop_source_ids else (doc.chunk_id,),
        )


__all__ = ["FakeMemory", "TransientFailure"]
