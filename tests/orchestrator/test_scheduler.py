"""The cross product, and the same-system lock that keeps latency honest (§3.5.1, §3.5.3)."""

import asyncio
import re

import pytest

from orchestrator.scheduler import Job, default_parallel, expand_jobs, run_jobs


def names(jobs: list[Job]) -> list[tuple[str, str]]:
    return [(j.system, j.dataset) for j in jobs]


def test_one_system_and_dataset_is_one_job() -> None:
    assert names(expand_jobs(["bm25"], ["locomo"])) == [("bm25", "locomo")]


def test_systems_and_datasets_expand_to_the_cross_product() -> None:
    jobs = expand_jobs(["a", "b"], ["x", "y"])
    assert names(jobs) == [("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")]


def test_every_job_gets_its_own_run_id() -> None:
    jobs = expand_jobs(["a", "b"], ["x"])
    assert len({j.run_id for j in jobs}) == len(jobs)


def test_a_run_id_is_generated_not_derived_from_the_names() -> None:
    """A run id is an opaque handle (§4.5); `bench ps` reads the names from the manifest."""
    job = expand_jobs(["bm25"], ["locomo"])[0]
    assert re.fullmatch(r"[0-9a-f]{12}", job.run_id)


def test_a_run_id_is_a_safe_directory_name() -> None:
    """It becomes a directory and part of a user_id, where `:` is a separator."""
    job = expand_jobs(["sys/a"], ["set:1"])[0]
    assert "/" not in job.run_id
    assert ":" not in job.run_id


async def test_jobs_run_to_completion() -> None:
    done: list[str] = []

    async def work(job: Job) -> None:
        done.append(job.run_id)

    jobs = expand_jobs(["a", "b"], ["x"])
    await run_jobs(jobs, work, parallel=2)
    assert sorted(done) == sorted(j.run_id for j in jobs)


async def test_runs_against_the_same_system_never_overlap() -> None:
    """Concurrent load on one container contaminates latency and rank stability."""
    active: dict[str, int] = {}
    peak = 0

    async def work(job: Job) -> None:
        nonlocal peak
        active[job.system] = active.get(job.system, 0) + 1
        peak = max(peak, active[job.system])
        await asyncio.sleep(0.01)
        active[job.system] -= 1

    await run_jobs(expand_jobs(["a"], ["x", "y", "z"]), work, parallel=3)
    assert peak == 1


async def test_runs_against_different_systems_do_overlap() -> None:
    started = asyncio.Event()
    both = asyncio.Event()
    seen = 0

    async def work(job: Job) -> None:
        nonlocal seen
        seen += 1
        if seen >= 2:
            both.set()
        started.set()
        await asyncio.wait_for(both.wait(), timeout=1.0)

    await run_jobs(expand_jobs(["a", "b"], ["x"]), work, parallel=2)
    assert both.is_set()


async def test_parallel_one_is_sequential() -> None:
    order: list[str] = []

    async def work(job: Job) -> None:
        order.append(f"start:{job.run_id}")
        await asyncio.sleep(0)
        order.append(f"end:{job.run_id}")

    first, second = expand_jobs(["a", "b"], ["x"])
    await run_jobs([first, second], work, parallel=1)
    assert order == [
        f"start:{first.run_id}",
        f"end:{first.run_id}",
        f"start:{second.run_id}",
        f"end:{second.run_id}",
    ]


async def test_the_same_system_override_allows_overlap() -> None:
    """3.5.1's escape hatch, which the report then marks `contended`."""
    active = 0
    peak = 0

    async def work(job: Job) -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1

    jobs = expand_jobs(["a"], ["x", "y"])
    await run_jobs(jobs, work, parallel=2, allow_concurrent_same_system=True)
    assert peak == 2


async def test_a_job_that_fails_does_not_stop_its_siblings() -> None:
    """A segfault or OOM in one run must not take down the others."""
    finished: list[str] = []

    async def work(job: Job) -> None:
        if job.system == "bad":
            raise RuntimeError("boom")
        finished.append(job.run_id)

    bad, good = expand_jobs(["bad", "good"], ["x"])
    results = await run_jobs([bad, good], work, parallel=2)
    assert finished == [good.run_id]
    assert results[bad.run_id] is not None
    assert results[good.run_id] is None


async def test_a_failure_is_reported_per_job() -> None:
    async def work(job: Job) -> None:
        raise RuntimeError("boom")

    job = expand_jobs(["a"], ["x"])[0]
    results = await run_jobs([job], work, parallel=1)
    assert "boom" in str(results[job.run_id])


async def test_cancellation_propagates() -> None:
    """A stage must let CancelledError bubble so SIGINT drains rather than hangs."""

    async def work(job: Job) -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_jobs(expand_jobs(["a"], ["x"]), work, parallel=1)


async def test_contended_jobs_are_marked_when_the_override_is_used() -> None:
    jobs = expand_jobs(["a"], ["x", "y"], contended=True)
    assert all(j.contended for j in jobs)


async def test_jobs_are_not_contended_by_default() -> None:
    assert not any(j.contended for j in expand_jobs(["a"], ["x"]))


def test_expand_jobs_accepts_ids_generated_by_a_launcher() -> None:
    """A detached batch's ids are printed before the supervisor exists, so it is handed them.

    Without this the supervisor would generate its own, and the ids the user was told to
    resume by would name nothing.
    """
    jobs = expand_jobs(["a", "b"], ["d1"], run_ids=["id-a", "id-b"])
    assert [job.run_id for job in jobs] == ["id-a", "id-b"]


def test_expand_jobs_still_generates_ids_when_given_none() -> None:
    jobs = expand_jobs(["a"], ["d1"])
    assert jobs[0].run_id


def test_expand_jobs_refuses_an_id_count_that_does_not_match() -> None:
    """Silently zipping short would drop jobs the user was already shown ids for."""
    with pytest.raises(ValueError, match="run_ids"):
        expand_jobs(["a", "b"], ["d1"], run_ids=["only-one"])


def test_default_parallelism_is_the_number_of_distinct_systems() -> None:
    """Same-system runs are serialized anyway, so the useful bound is distinct systems.

    ARCH §"Parallel runs" calls that the bound on useful parallelism; defaulting below it
    made `bench run a,b` serial for no fairness benefit.
    """
    assert default_parallel([Job("r1", "a", "d"), Job("r2", "b", "d")]) == 2


def test_repeated_systems_do_not_raise_the_default() -> None:
    """Two datasets on one system still serialize, so a higher cap would buy nothing."""
    assert default_parallel([Job("r1", "a", "d1"), Job("r2", "a", "d2")]) == 1


def test_default_parallelism_is_never_zero() -> None:
    """An empty batch cannot produce a semaphore of 0, which would deadlock."""
    assert default_parallel([]) == 1
