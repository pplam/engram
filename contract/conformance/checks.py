"""Contract checks runnable against any adapter (ARCH §8, §1.5).

Each check is an async callable over a `MemoryAdapter`. It raises `CheckFailed` with a
reason, or returns. `run_all` turns them into results so a caller sees every check even
after one fails; `bench doctor` and the pytest conformance suite both go through here,
so there is one definition per row.

Two rows from the previous HTTP contract are gone with the endpoints they tested:
`health` (there is no endpoint; a failing `add` is the signal) and `auth_accepted` (there
is no harness-supplied credential — an adapter that cannot authenticate fails to
construct, which the registry already reports).
"""

import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from contract.adapter import Memory, MemoryAdapter, Message
from orchestrator.ids import user_id

DATASET = "conformance"

# Probes are fact-shaped sentences carrying a nonce, and every check matches on the nonce
# rather than on the sentence.
#
# Both halves are required by an extractive system. mem0 runs an LLM over the messages and
# stores what it judges worth keeping, paraphrased: `add`ing "quokka a1b2c3d4 sandwich"
# returns `{"results": []}` and stores nothing, while "My locker number is a1b2c3d4."
# stores "User's locker number is a1b2c3d4." So a probe that is not a fact tests nothing,
# and a verbatim assertion over one that is fails on the rewrite. The nonce is what
# survives, and it is what a system cannot produce without having stored the probe.
_CANARY_SUBJECT = "safe-deposit box"


def _fact(subject: str, nonce: str) -> str:
    """Return a fact-shaped statement about `subject` carrying `nonce`."""
    return f"My {subject} number is {nonce}."


class CheckFailed(Exception):
    """A contract check did not hold."""


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check."""

    name: str
    passed: bool
    reason: str | None = None


@dataclass(frozen=True)
class Check:
    """A named contract check."""

    name: str
    description: str
    run: Callable[[MemoryAdapter, str], Awaitable[None]]


async def _add(adapter: MemoryAdapter, uid: str, content: str, chunk_id: str) -> None:
    message = Message(role="user", content=content, timestamp=1704067200000)
    await adapter.add(uid, [message], chunk_id)


async def _search(adapter: MemoryAdapter, uid: str, query: str, top_k: int) -> Sequence[Memory]:
    found = await adapter.search(uid, query, top_k)
    for memory in found:
        if not isinstance(memory, Memory):
            raise CheckFailed(f"search returned a {type(memory).__name__}, expected a Memory")
    return found


async def check_adapter_constructs(adapter: MemoryAdapter, run_id: str) -> None:
    """The adapter was importable, constructible, and satisfies the protocol."""
    for method in ("add", "search"):
        if not callable(getattr(adapter, method, None)):
            raise CheckFailed(f"adapter has no callable {method}")


async def check_searchable_on_return(adapter: MemoryAdapter, run_id: str) -> None:
    """Content stored by a returned `add` is retrievable on the next `search`.

    An adapter that returns before its system has indexed gets flagged `late_indexing`,
    which marks the run's results provisional — the questions retrieved early saw less
    of the corpus than the ones retrieved late.
    """
    uid = user_id(run_id, DATASET, "searchable")
    nonce = uuid.uuid4().hex[:8]
    probe = _fact("locker", nonce)
    await _add(adapter, uid, probe, "c0")
    found = await _search(adapter, uid, probe, top_k=10)
    if not any(nonce in m.content for m in found):
        raise CheckFailed("content stored by a returned add was not searchable")


async def check_top_k_honoured(adapter: MemoryAdapter, run_id: str) -> None:
    """A search returns at most `top_k` memories."""
    uid = user_id(run_id, DATASET, "topk")
    for i in range(6):
        await _add(adapter, uid, _fact("budget", f"{i}{uuid.uuid4().hex[:6]}"), f"c{i}")
    top_k = 3
    found = await _search(adapter, uid, "my budget number", top_k=top_k)
    if len(found) > top_k:
        raise CheckFailed(f"returned {len(found)} memories for top_k={top_k}")


async def check_content_present(adapter: MemoryAdapter, run_id: str) -> None:
    """Every returned memory carries non-empty content.

    Empty content would be scored as a wrong answer, which reads as a memory result
    rather than as the broken integration it is.
    """
    uid = user_id(run_id, DATASET, "content")
    await _add(adapter, uid, _fact("cat's collar tag", uuid.uuid4().hex[:8]), "c0")
    found = await _search(adapter, uid, "my cat's collar tag number", top_k=10)
    if any(not m.content for m in found):
        raise CheckFailed("a returned memory has empty content")


async def check_no_cross_user_leakage(adapter: MemoryAdapter, run_id: str) -> None:
    """A canary stored under a sibling `user_id` is not retrievable by the target.

    `user_id` is the isolation boundary, and this is the one violation that invalidates
    a whole run rather than one row: a system that can see other users' content is not
    answering from memory, it is answering from the corpus.

    Because of that, the canary is confirmed present under the sibling before absence
    under the target is read as isolation. Nothing leaks out of an empty store, so
    without that step a system which silently dropped the canary would earn the
    strongest claim the suite makes by storing nothing at all.
    """
    target = user_id(run_id, DATASET, "leak-target")
    sibling = user_id(run_id, DATASET, "leak-sibling")
    nonce = uuid.uuid4().hex[:8]
    canary = _fact(_CANARY_SUBJECT, nonce)

    await _add(adapter, sibling, canary, "c0")
    await _add(adapter, target, _fact("bicycle lock", uuid.uuid4().hex[:8]), "c0")

    # Searched twice, because a late-indexing system reveals content on a later read and
    # that is `searchable_on_return`'s finding to report, not this check's. What remains
    # after the retry is a store that does not hold the canary at all.
    stored = await _search(adapter, sibling, canary, top_k=10)
    if not any(nonce in m.content for m in stored):
        stored = await _search(adapter, sibling, canary, top_k=10)
    if not any(nonce in m.content for m in stored):
        raise CheckFailed(
            "inconclusive: the canary was not retrievable under the user_id that stored it, "
            "so its absence under a sibling shows nothing about isolation"
        )

    found = await _search(adapter, target, canary, top_k=10)
    if any(nonce in m.content for m in found):
        raise CheckFailed("a canary from a sibling user_id leaked into the target user's search")


CHECKS: tuple[Check, ...] = (
    Check("adapter_constructs", "adapter satisfies MemoryAdapter", check_adapter_constructs),
    Check("searchable_on_return", "add returns only once searchable", check_searchable_on_return),
    Check("top_k_honoured", "len(memories) <= top_k", check_top_k_honoured),
    Check("content_present", "every memory has content", check_content_present),
    Check(
        "no_cross_user_leakage",
        "sibling user_id canary not retrievable",
        check_no_cross_user_leakage,
    ),
)


async def run_all(adapter: MemoryAdapter, run_id: str) -> list[CheckResult]:
    """Run every check, returning one result each; never raises for a failed check."""
    results: list[CheckResult] = []
    for check in CHECKS:
        try:
            await check.run(adapter, run_id)
        except CheckFailed as err:
            results.append(CheckResult(check.name, passed=False, reason=str(err)))
        except Exception as err:  # noqa: BLE001 - a third-party adapter may raise anything
            results.append(
                CheckResult(check.name, passed=False, reason=f"{type(err).__name__}: {err}")
            )
        else:
            results.append(CheckResult(check.name, passed=True))
    return results


__all__ = ["CHECKS", "Check", "CheckFailed", "CheckResult", "run_all"]
