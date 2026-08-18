"""The acceptance test for Phase 3.5: parallelism must not change artifacts.

"A parallel run is byte-identical to a solo run. Any concurrency that changes artifacts
is a bug, not a tradeoff." Everything else in the phase is a convenience; this is the
property that makes it safe.
"""

import shutil
from pathlib import Path

import pytest

from contract.adapter import MemoryAdapter
from orchestrator.models import StubChat
from orchestrator.run import RunRequest, execute_run
from orchestrator.scheduler import Job, expand_jobs, run_jobs
from reference.baselines import build_baseline

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "datasets" / "fixtures"

# Artifacts whose bytes must not depend on how the run was scheduled. `ingest.jsonl` and
# `retrieve.jsonl` carry per-call latencies and wall-clock timestamps, and `state.json`
# carries a pid, so those legitimately differ; the scored output must not.
DETERMINISTIC = ("plan.jsonl", "manifest.json", "answer.jsonl", "judge.jsonl")


@pytest.fixture
def datasets(tmp_path: Path) -> Path:
    dest = tmp_path / "datasets"
    shutil.copytree(FIXTURES, dest)
    return dest


def adapter_for(system: str) -> MemoryAdapter:
    """A fresh adapter per job: two concurrent jobs must not share one system's state."""
    return build_baseline(system)


def stub() -> StubChat:
    return StubChat(replies=['{"label": "CORRECT"}'], prompt_tokens=7, completion_tokens=3)


async def execute(job: Job, datasets: Path, artifacts: Path) -> None:
    request = RunRequest(
        run_id=job.run_id,
        system=job.system,
        dataset=job.dataset,
        suite="v1",
        artifacts_root=artifacts,
        datasets_root=datasets,
        repo_root=REPO,
        contended=job.contended,
    )
    await execute_run(request, adapter_for(job.system), chat=stub(), judge_chat=stub())


def read_all(run_dir: Path) -> dict[str, bytes]:
    return {name: (run_dir / name).read_bytes() for name in DETERMINISTIC}


async def test_a_parallel_run_is_byte_identical_to_a_solo_run(
    datasets: Path, tmp_path: Path
) -> None:
    solo_root = tmp_path / "solo"
    jobs = expand_jobs(["bm25", "no_memory"], ["fixture_many"])
    for job in jobs:
        await execute(job, datasets, solo_root)

    parallel_root = tmp_path / "parallel"
    results = await run_jobs(jobs, lambda job: execute(job, datasets, parallel_root), parallel=2)
    assert all(err is None for err in results.values()), results

    for job in jobs:
        assert read_all(solo_root / job.run_id) == read_all(parallel_root / job.run_id), job.run_id


async def test_parallel_runs_do_not_share_a_namespace(datasets: Path, tmp_path: Path) -> None:
    """`user_id` embeds `run_id`, which is what makes runs independent (§4.4)."""
    artifacts = tmp_path / "artifacts"
    jobs = expand_jobs(["bm25", "no_memory"], ["fixture_many"])
    await run_jobs(jobs, lambda job: execute(job, datasets, artifacts), parallel=2)

    from orchestrator.artifacts import JsonlArtifact

    namespaces = {
        job.run_id: {
            str(row["user_id"])
            for row in JsonlArtifact(artifacts / job.run_id / "retrieve.jsonl").stream()
        }
        for job in jobs
    }
    first, second = (namespaces[job.run_id] for job in jobs)
    assert first and second
    assert not (first & second)


async def test_each_parallel_run_writes_its_own_report(datasets: Path, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    jobs = expand_jobs(["bm25", "no_memory"], ["fixture_many"])
    await run_jobs(jobs, lambda job: execute(job, datasets, artifacts), parallel=2)
    for job in jobs:
        assert (artifacts / job.run_id / "report.json").is_file()


async def test_serialized_same_system_runs_still_produce_distinct_artifacts(
    datasets: Path, tmp_path: Path
) -> None:
    """Two runs of one system are queued, not merged; each keeps its own directory."""
    artifacts = tmp_path / "artifacts"
    jobs = [
        Job(run_id="first", system="bm25", dataset="fixture_many"),
        Job(run_id="second", system="bm25", dataset="fixture_many"),
    ]
    await run_jobs(jobs, lambda job: execute(job, datasets, artifacts), parallel=2)
    assert (artifacts / "first" / "report.json").is_file()
    assert (artifacts / "second" / "report.json").is_file()
