"""Tests-of-tests: each flawed fake must fail exactly the checks it breaks (§1.5)."""

from collections.abc import Callable

import pytest

from contract.conformance.checks import CHECKS, CheckFailed, run_all
from reference.fakes import FakeMemory


def test_every_check_has_a_unique_name() -> None:
    names = [c.name for c in CHECKS]
    assert len(names) == len(set(names))


def test_the_suite_covers_the_verifiable_contract_rows() -> None:
    assert {c.name for c in CHECKS} == {
        "adapter_constructs",
        "searchable_on_return",
        "top_k_honoured",
        "content_present",
        "no_cross_user_leakage",
    }


async def test_a_conformant_fake_passes_every_check() -> None:
    results = await run_all(FakeMemory(), run_id="t-good")
    assert [r.name for r in results if not r.passed] == []


@pytest.mark.parametrize(
    ("flawed", "expected_failures"),
    [
        (lambda: FakeMemory(leak_across_users=True), {"no_cross_user_leakage"}),
        (lambda: FakeMemory(overshoot_top_k=True), {"top_k_honoured"}),
        (lambda: FakeMemory(index_late=True), {"searchable_on_return"}),
        # Empty content breaks three properties, not one: content that is not there is
        # also not searchable, and it cannot carry the canary the isolation check has to
        # find before absence under a sibling means anything. The overlap is in the fake,
        # not in the checks.
        (
            lambda: FakeMemory(empty_content=True),
            {"content_present", "searchable_on_return", "no_cross_user_leakage"},
        ),
    ],
    ids=["leak_across_users", "overshoot_top_k", "index_late", "empty_content"],
)
async def test_each_flaw_fails_the_checks_it_breaks(
    flawed: Callable[[], FakeMemory], expected_failures: set[str]
) -> None:
    results = await run_all(flawed(), run_id="t-flaw")
    assert {r.name for r in results if not r.passed} == expected_failures


async def test_an_extractive_system_passes_every_check() -> None:
    """Storing derived facts rather than raw text is conformant, not a violation.

    mem0 is extractive: `add` runs an LLM over the messages and stores what it judges
    worth keeping, paraphrased. Probes must therefore be fact-shaped and matched on a
    nonce that survives the rewrite — a check that only a verbatim store can pass would
    measure storage style instead of the contract.
    """
    results = await run_all(FakeMemory(extractive=True), run_id="t-extractive")
    assert [r.name for r in results if not r.passed] == []


async def test_a_system_that_stores_nothing_cannot_pass_the_isolation_check() -> None:
    """Isolation is only demonstrated once the canary is known to have been stored.

    Otherwise the check passes for the wrong reason: nothing leaked because nothing
    exists. This is the one failure that invalidates a whole run, so a vacuous PASS on
    it is worse than a FAIL — it is an unearned claim. Against real mem0 the old
    nonsense canary stored nothing and the row read PASS.
    """
    results = await run_all(FakeMemory(store_nothing=True), run_id="t-void")
    failed = {r.name for r in results if not r.passed}
    assert "no_cross_user_leakage" in failed


async def test_the_isolation_check_says_when_it_could_not_conclude() -> None:
    """An empty store and a leak are different bugs, so the reason must say which."""
    results = await run_all(FakeMemory(store_nothing=True), run_id="t-void-reason")
    leakage = next(r for r in results if r.name == "no_cross_user_leakage")
    assert leakage.reason is not None
    assert "leaked" not in leakage.reason


async def test_missing_source_ids_is_not_a_contract_violation() -> None:
    """`source_ids` are optional: without them recall falls back to the overlap track."""
    results = await run_all(FakeMemory(drop_source_ids=True), run_id="t-src")
    assert [r.name for r in results if not r.passed] == []


async def test_failure_carries_a_reason_naming_the_problem() -> None:
    results = await run_all(FakeMemory(overshoot_top_k=True), run_id="t-reason")
    failure = next(r for r in results if not r.passed)
    assert failure.reason is not None
    assert "top_k" in failure.reason


async def test_passing_result_has_no_reason() -> None:
    results = await run_all(FakeMemory(), run_id="t-clean")
    assert all(r.reason is None for r in results)


async def test_run_all_reports_every_check_even_when_one_fails() -> None:
    results = await run_all(FakeMemory(index_late=True), run_id="t-all")
    assert len(results) == len(CHECKS)


async def test_an_adapter_that_raises_fails_its_checks_rather_than_raising() -> None:
    """A third-party adapter may raise anything; `doctor` still has to print a table."""
    results = await run_all(FakeMemory(fail_adds=99), run_id="t-raise")
    failed = {r.name for r in results if not r.passed}
    assert failed == {
        "searchable_on_return",
        "top_k_honoured",
        "content_present",
        "no_cross_user_leakage",
    }


async def test_check_failed_is_raised_with_the_offending_detail() -> None:
    with pytest.raises(CheckFailed, match="boom"):
        raise CheckFailed("boom")
