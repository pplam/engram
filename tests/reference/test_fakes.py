"""The fakes are how the conformance suite is tested (§1.4).

Each misbehaving fake carries exactly one flag, so a conformance check that fires on
the wrong fake is a bug in the check rather than in the fake. None of them needs a
socket: tests instantiate the class.
"""

import pytest

from contract.adapter import Memory, MemoryAdapter, Message
from reference.fakes import FakeMemory

UID = "eval:r1:fixture:ctx0"
OTHER = "eval:r1:fixture:ctx1"


def msg(content: str) -> list[Message]:
    return [Message(role="user", content=content)]


@pytest.fixture
def good() -> FakeMemory:
    return FakeMemory()


async def test_the_conformant_fake_satisfies_the_protocol(good: FakeMemory) -> None:
    assert isinstance(good, MemoryAdapter)


async def test_stored_content_is_searchable_on_return(good: FakeMemory) -> None:
    await good.add(UID, msg("the cat sat on the mat"), chunk_id="c0")
    found = await good.search(UID, "cat", top_k=5)
    assert [m.content for m in found] == ["the cat sat on the mat"]


async def test_search_honours_top_k(good: FakeMemory) -> None:
    for i in range(6):
        await good.add(UID, msg(f"probe number {i}"), chunk_id=f"c{i}")
    assert len(await good.search(UID, "probe", top_k=3)) == 3


async def test_source_ids_carry_the_chunk_id(good: FakeMemory) -> None:
    """Without this the row scores quality but recall reads `unavailable`."""
    await good.add(UID, msg("a cat named Mochi"), chunk_id="c7")
    assert list((await good.search(UID, "cat", top_k=5))[0].source_ids) == ["c7"]


async def test_user_id_isolates_content(good: FakeMemory) -> None:
    await good.add(OTHER, msg("canary belonging to a sibling"), chunk_id="c0")
    assert await good.search(UID, "canary", top_k=5) == []


async def test_a_query_matching_nothing_returns_empty(good: FakeMemory) -> None:
    await good.add(UID, msg("the cat sat"), chunk_id="c0")
    assert await good.search(UID, "aardvark", top_k=5) == []


async def test_results_come_back_best_first(good: FakeMemory) -> None:
    """Rank order is the position in the sequence, so it has to be meaningful."""
    await good.add(UID, msg("cat"), chunk_id="c0")
    await good.add(UID, msg("cat cat cat"), chunk_id="c1")
    found = await good.search(UID, "cat", top_k=5)
    assert [list(m.source_ids) for m in found] == [["c1"], ["c0"]]


async def test_leaks_across_users_returns_a_siblings_content() -> None:
    leaky = FakeMemory(leak_across_users=True)
    await leaky.add(OTHER, msg("canary belonging to a sibling"), chunk_id="c0")
    found = await leaky.search(UID, "canary", top_k=5)
    assert [m.content for m in found] == ["canary belonging to a sibling"]


async def test_overshoots_top_k_returns_more_than_asked() -> None:
    greedy = FakeMemory(overshoot_top_k=True)
    for i in range(20):
        await greedy.add(UID, msg(f"probe number {i}"), chunk_id=f"c{i}")
    assert len(await greedy.search(UID, "probe", top_k=3)) == 8


async def test_empty_content_returns_memories_with_no_text() -> None:
    hollow = FakeMemory(empty_content=True)
    await hollow.add(UID, msg("the cat sat on the mat"), chunk_id="c0")
    found = await hollow.search(UID, "cat", top_k=5)
    assert found and all(m.content == "" for m in found)


async def test_drops_source_ids_reports_none() -> None:
    anonymous = FakeMemory(drop_source_ids=True)
    await anonymous.add(UID, msg("a cat named Mochi"), chunk_id="c7")
    found = await anonymous.search(UID, "cat", top_k=5)
    assert found and all(m.source_ids == () for m in found)


async def test_indexes_late_is_invisible_until_the_second_search() -> None:
    """`add` returning before content is searchable is the `late_indexing` failure."""
    slow = FakeMemory(index_late=True)
    await slow.add(UID, msg("the cat sat on the mat"), chunk_id="c0")
    assert await slow.search(UID, "cat", top_k=5) == []
    assert [m.content for m in await slow.search(UID, "cat", top_k=5)] == ["the cat sat on the mat"]


async def test_a_fake_can_raise_a_retryable_error_a_fixed_number_of_times() -> None:
    """Exercises the retry policy without a socket or a clock."""
    flaky = FakeMemory(fail_adds=2)
    with pytest.raises(Exception, match="transient"):
        await flaky.add(UID, msg("x"), chunk_id="c0")
    with pytest.raises(Exception, match="transient"):
        await flaky.add(UID, msg("x"), chunk_id="c0")
    await flaky.add(UID, msg("the cat sat"), chunk_id="c0")
    assert len(await flaky.search(UID, "cat", top_k=5)) == 1


async def test_search_returns_memory_records(good: FakeMemory) -> None:
    await good.add(UID, msg("the cat sat"), chunk_id="c0")
    assert all(isinstance(m, Memory) for m in await good.search(UID, "cat", top_k=5))
