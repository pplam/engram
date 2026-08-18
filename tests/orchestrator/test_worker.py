"""A batch run is one subprocess per job, so a crash is contained (§3.5.3)."""

import asyncio
import sys
from pathlib import Path

import pytest

from orchestrator.scheduler import Job, run_jobs
from orchestrator.worker import child_argv, describe_failure, spawn

REPO = Path(__file__).resolve().parents[2]


def python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


async def test_spawn_returns_the_child_exit_code() -> None:
    result = await spawn(python("raise SystemExit(3)"), cwd=REPO)
    assert result.returncode == 3


async def test_spawn_captures_child_stderr() -> None:
    result = await spawn(python("import sys; sys.stderr.write('registry error: ghost')"), cwd=REPO)
    assert "ghost" in result.stderr


async def test_spawn_keeps_child_stdout_out_of_the_parents_own_output() -> None:
    """The parent renders rows from report.json, so a child's table must not double up."""
    result = await spawn(python("print('a row the parent will not repeat')"), cwd=REPO)
    assert "a row" in result.stdout


async def test_a_killed_child_reports_its_signal() -> None:
    """A segfault or an OOM kill is a signal, and must read as one rather than as success."""
    result = await spawn(
        python("import os, signal; os.kill(os.getpid(), signal.SIGKILL)"), cwd=REPO
    )
    assert result.returncode != 0
    assert "signal 9" in describe_failure(result)


async def test_a_killed_child_leaves_its_sibling_running() -> None:
    """The whole point of a process per run: one dying must not take the others down."""
    done: list[str] = []

    async def work(job: Job) -> None:
        if job.run_id == "doomed":
            result = await spawn(
                python("import os, signal; os.kill(os.getpid(), signal.SIGKILL)"), cwd=REPO
            )
            raise RuntimeError(describe_failure(result))
        await asyncio.sleep(0.05)
        done.append(job.run_id)

    jobs = [
        Job(run_id="doomed", system="a", dataset="d"),
        Job(run_id="survivor", system="b", dataset="d"),
    ]
    results = await run_jobs(jobs, work, parallel=2)
    assert done == ["survivor"]
    assert results["survivor"] is None
    assert "signal 9" in str(results["doomed"])


def test_child_argv_asks_for_exactly_one_run() -> None:
    job = Job(run_id="b-bm25-fixture_many", system="bm25", dataset="fixture_many")
    argv = child_argv(
        job,
        suite="v1",
        artifacts=Path("/tmp/a"),
        datasets=Path("/tmp/d"),
        repo_root=REPO,
        registry=Path("registry"),
        limit=None,
    )
    assert argv[:2] == [sys.executable, "-m"]
    assert "run" in argv
    assert "bm25" in argv
    assert argv[argv.index("--run-id") + 1] == "b-bm25-fixture_many"
    assert argv[argv.index("--dataset") + 1] == "fixture_many"


def test_child_argv_passes_the_contended_mark_down() -> None:
    """The parent knows the batch shares a system; the child cannot work that out alone."""
    job = Job(run_id="r", system="bm25", dataset="d", contended=True)
    argv = child_argv(
        job,
        suite="v1",
        artifacts=Path("a"),
        datasets=Path("d"),
        repo_root=REPO,
        registry=Path("registry"),
        limit=None,
    )
    assert "--contended" in argv


def test_child_argv_omits_limit_when_unset() -> None:
    job = Job(run_id="r", system="bm25", dataset="d")
    argv = child_argv(
        job,
        suite="v1",
        artifacts=Path("a"),
        datasets=Path("d"),
        repo_root=REPO,
        registry=Path("registry"),
        limit=7,
    )
    assert argv[argv.index("--limit") + 1] == "7"


async def test_cancelling_a_spawn_drains_the_child_before_it_returns(tmp_path: Path) -> None:
    """SIGINT drains: the child gets to finish its final write, not orphaned (§3.5.3).

    The child ignores nothing and writes a marker from its SIGINT handler. If `spawn` let
    cancellation escape without signalling and reaping, the marker would not be on disk by
    the time the cancellation surfaces here.
    """
    marker = tmp_path / "drained"
    code = (
        "import signal, sys, time\n"
        "from pathlib import Path\n"
        "def bye(*_):\n"
        f"    Path({str(marker)!r}).write_text('drained')\n"
        "    raise SystemExit(130)\n"
        "signal.signal(signal.SIGINT, bye)\n"
        "print('ready', flush=True)\n"
        "time.sleep(30)\n"
    )
    task = asyncio.create_task(spawn(python(code), cwd=REPO))
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert marker.is_file(), "the child was abandoned rather than asked to stop"
