"""Expanding the cross product and running it under two bounds (§3.5.1, §3.5.3).

Runs are independent by construction — `user_id` embeds `run_id`, artifacts live in
per-run directories, and `score` is pure — so parallelism across different systems needs
no new isolation. Two things are genuinely shared and are bounded here:

- **A single system under test.** Runs against the same system are **serialized**. This
  is a fairness requirement, not a performance choice: §7.4 reports latency and §7.5
  reports rank stability, and two concurrent runs against one container contaminate both.
  `--allow-concurrent-same-system` overrides it, and those runs are marked `contended`
  so their numbers cannot be quoted as clean.
- **Total concurrency**, capped by `parallel`.

A failing job must not take its siblings down, so failures are collected per job rather
than propagated. `CancelledError` is deliberately *not* caught: SIGINT has to drain, and
in-flight runs are resumable by ID anyway.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from orchestrator.ids import new_run_id


@dataclass(frozen=True)
class Job:
    """One `(system, dataset)` run in a batch."""

    run_id: str
    system: str
    dataset: str
    contended: bool = False


def expand_jobs(
    systems: list[str],
    datasets: list[str],
    contended: bool = False,
    run_ids: list[str] | None = None,
) -> list[Job]:
    """Return one job per `(system, dataset)` pair, in a deterministic order.

    Each job gets a generated run id (§4.5). Ids are no longer derived from the system
    and dataset names: a run id is an opaque handle, and `bench ps` reads what a run *is*
    from its manifest rather than from its directory name.

    `run_ids` supplies them instead, in this same order, for a detached batch: the
    launcher prints the ids and exits, so the supervisor that actually runs the jobs has
    to be handed the ids the user was already shown rather than inventing new ones.
    """
    pairs = [(system, dataset) for system in systems for dataset in datasets]
    if run_ids is None:
        ids = [new_run_id() for _ in pairs]
    elif len(run_ids) != len(pairs):
        # Zipping short would silently drop jobs the user has already been given ids for.
        raise ValueError(
            f"expected {len(pairs)} run_ids for this cross product, got {len(run_ids)}"
        )
    else:
        ids = list(run_ids)

    return [
        Job(run_id=run_id, system=system, dataset=dataset, contended=contended)
        for run_id, (system, dataset) in zip(ids, pairs, strict=True)
    ]


def default_parallel(jobs: list[Job]) -> int:
    """Return how many of `jobs` may usefully run at once: the distinct system count.

    Runs against one system are serialized regardless (see `run_jobs`), so a cap above the
    number of distinct systems buys nothing, and a cap below it leaves systems idle for no
    fairness benefit — which is what a hardcoded default of 1 did to every cross product.
    Never zero: an empty batch must not build a semaphore that blocks forever.
    """
    return max(len({job.system for job in jobs}), 1)


async def run_jobs(
    jobs: list[Job],
    work: Callable[[Job], Awaitable[None]],
    parallel: int,
    allow_concurrent_same_system: bool = False,
) -> dict[str, BaseException | None]:
    """Run every job, returning each one's failure or None. Never raises for a job failure."""
    overall = asyncio.Semaphore(max(parallel, 1))
    # One lock per system is what serializes runs against the same container.
    locks: dict[str, asyncio.Lock] = {job.system: asyncio.Lock() for job in jobs}
    results: dict[str, BaseException | None] = {}

    async def run_one(job: Job) -> None:
        async with overall:
            if allow_concurrent_same_system:
                await _guarded(job, work, results)
            else:
                async with locks[job.system]:
                    await _guarded(job, work, results)

    # A bare gather would let one job's failure cancel the rest, so each job captures
    # its own outcome and only cancellation escapes.
    await asyncio.gather(*(run_one(job) for job in jobs))
    return results


async def _guarded(
    job: Job,
    work: Callable[[Job], Awaitable[None]],
    results: dict[str, BaseException | None],
) -> None:
    try:
        await work(job)
    except asyncio.CancelledError:
        # Cancellation is not a job failure; let it drain the whole batch.
        raise
    except Exception as err:  # noqa: BLE001 - one run's crash must not stop its siblings
        results[job.run_id] = err
    else:
        results[job.run_id] = None
