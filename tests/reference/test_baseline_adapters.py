"""`no_memory` and `bm25` as ordinary adapters (ARCH §10).

They are the cheapest check that the interface is implementable: they satisfy the same
protocol a third party would, need no registry entry, and pass conformance unchanged.
`bm25` is also the shortest complete example of an adapter that reports `source_ids`.
"""

import pytest

from contract.adapter import MemoryAdapter, Message
from reference.baselines import BaselineAdapter, Bm25Adapter, NoMemoryAdapter, build_baseline

UID = "eval:r1:fixture:ctx0"
OTHER = "eval:r1:fixture:ctx1"


def msg(content: str) -> list[Message]:
    return [Message(role="user", content=content)]


@pytest.mark.parametrize("adapter", [NoMemoryAdapter(), Bm25Adapter()])
def test_both_baselines_satisfy_the_protocol(adapter: BaselineAdapter) -> None:
    assert isinstance(adapter, MemoryAdapter)


async def test_no_memory_returns_nothing_it_was_given() -> None:
    """The floor for every dataset: accepts every write and forgets it."""
    floor = NoMemoryAdapter()
    await floor.add(UID, msg("the cat sat on the mat"), chunk_id="c0")
    assert not await floor.search(UID, "cat", top_k=5)


async def test_bm25_retrieves_what_it_stored() -> None:
    lexical = Bm25Adapter()
    await lexical.add(UID, msg("Ana opened a bakery in Porto"), chunk_id="c1")
    await lexical.add(UID, msg("unrelated talk about the weather"), chunk_id="c2")
    found = await lexical.search(UID, "Ana bakery Porto", top_k=5)
    assert list(found[0].source_ids) == ["c1"]


async def test_bm25_reports_source_ids() -> None:
    lexical = Bm25Adapter()
    await lexical.add(UID, msg("a cat named Mochi"), chunk_id="c7")
    assert list((await lexical.search(UID, "cat", top_k=5))[0].source_ids) == ["c7"]


async def test_bm25_honours_top_k() -> None:
    lexical = Bm25Adapter()
    for i in range(6):
        await lexical.add(UID, msg(f"probe number {i}"), chunk_id=f"c{i}")
    assert len(await lexical.search(UID, "probe number", top_k=3)) == 3


async def test_bm25_isolates_by_user_id() -> None:
    lexical = Bm25Adapter()
    await lexical.add(OTHER, msg("canary belonging to a sibling"), chunk_id="c0")
    assert await lexical.search(UID, "canary", top_k=5) == []


async def test_bm25_ranks_by_score_descending() -> None:
    lexical = Bm25Adapter()
    await lexical.add(UID, msg("zebra"), chunk_id="c9")
    await lexical.add(UID, msg("the common zebra grazes"), chunk_id="c8")
    found = await lexical.search(UID, "common zebra", top_k=5)
    # `score` is optional on the interface, so a ranking claim needs it present first.
    scores = [m.score for m in found]
    assert all(s is not None for s in scores)
    ranked = [s for s in scores if s is not None]
    assert ranked == sorted(ranked, reverse=True)
    assert list(found[0].source_ids) == ["c8"]


async def test_bm25_indexes_one_document_per_message() -> None:
    """Chunks hold several messages; ranking them separately is the point of bm25."""
    lexical = Bm25Adapter()
    await lexical.add(
        UID,
        [Message(role="user", content="first about cats"), Message(role="user", content="second")],
        chunk_id="c0",
    )
    assert len(await lexical.search(UID, "first second", top_k=5)) == 2


@pytest.mark.parametrize("name", ["no_memory", "bm25"])
def test_build_baseline_returns_an_adapter_by_name(name: str) -> None:
    assert isinstance(build_baseline(name), MemoryAdapter)


def test_build_baseline_rejects_an_unknown_name() -> None:
    with pytest.raises(KeyError, match="oracle_gold"):
        build_baseline("oracle_gold")
