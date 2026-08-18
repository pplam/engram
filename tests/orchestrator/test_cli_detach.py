"""`bench run` returns immediately and the run continues in the background (§3.5.3).

A run takes hours. Holding the launching terminal for it means a closed laptop lid or a
stray Ctrl-C ends the run, so detaching is the default and `--foreground` is the opt-in
for the old streaming behaviour.
"""

import shutil
import time
from pathlib import Path

import pytest

from orchestrator.cli import main
from orchestrator.runlog import RUN_LOG
from orchestrator.runstate import read_state, resolve_status

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "datasets" / "fixtures"


@pytest.fixture
def datasets(tmp_path: Path) -> Path:
    dest = tmp_path / "datasets"
    shutil.copytree(FIXTURES, dest)
    return dest


def argv(datasets: Path, artifacts: Path, *extra: str) -> list[str]:
    """A run that needs no model call, so a detached child needs no gateway.

    `conftest`'s offline transport patches this process, and a detached run is a different
    one — it would reach for a real endpoint. `--limit 0` plans no questions, so `answer`
    and `judge` have nothing to call, and what these tests assert on (detaching, the pid,
    the log, the report) is unaffected.
    """
    return [
        "run",
        "bm25",
        "--limit",
        "0",
        "--dataset",
        "fixture_many",
        "--suite",
        "v1",
        "--artifacts",
        str(artifacts),
        "--datasets",
        str(datasets),
        "--repo-root",
        str(REPO),
        *extra,
    ]


def await_run(artifacts: Path, run_id: str, timeout_s: float = 60.0) -> str:
    """Block until the detached run leaves `running`, then return its status.

    Polling rather than reading once: the launcher returns before the child has written
    anything, which is the whole point of detaching.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = read_state(artifacts / run_id)
        if state is not None and state.status in ("done", "failed"):
            return resolve_status(state)
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish within {timeout_s}s")


def test_run_detaches_and_prints_only_the_run_id(
    datasets: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifacts = tmp_path / "artifacts"
    assert main(argv(datasets, artifacts)) == 0
    out = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    # Exactly one line, and it is the id. The run has not finished, so there is no row
    # to print and nothing may claim otherwise.
    assert len(out) == 1, out
    run_id = out[0].strip()
    assert len(run_id) == 12, out
    assert await_run(artifacts, run_id) == "done"


def launch(datasets: Path, artifacts: Path, capsys: pytest.CaptureFixture[str]) -> str:
    """Launch a detached run and return the id the launcher printed."""
    assert main(argv(datasets, artifacts)) == 0
    return capsys.readouterr().out.strip().splitlines()[0].strip()


def test_the_detached_run_is_a_separate_process(
    datasets: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A detached run must outlive the launching shell, so it cannot share its pid."""
    import os

    artifacts = tmp_path / "artifacts"
    run_id = launch(datasets, artifacts, capsys)
    await_run(artifacts, run_id)
    state = read_state(artifacts / run_id)
    assert state is not None
    assert state.pid != os.getpid()


def test_the_detached_run_writes_its_log_and_report(
    datasets: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifacts = tmp_path / "artifacts"
    run_id = launch(datasets, artifacts, capsys)
    assert await_run(artifacts, run_id) == "done"
    assert (artifacts / run_id / "report.json").is_file()
    assert "run done" in (artifacts / run_id / RUN_LOG).read_text()


def test_foreground_still_prints_the_report_row(
    datasets: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv(datasets, tmp_path / "artifacts", "--foreground")) == 0
    out = capsys.readouterr().out
    assert "bm25" in out
    assert "fixture_many" in out


def test_foreground_stays_in_the_launching_process(datasets: Path, tmp_path: Path) -> None:
    import os

    artifacts = tmp_path / "artifacts"
    main(argv(datasets, artifacts, "--foreground"))
    run_id = next(p.name for p in artifacts.iterdir() if (p / "state.json").is_file())
    state = read_state(artifacts / run_id)
    assert state is not None
    assert state.pid == os.getpid()


def test_detach_reports_a_usage_error_without_spawning(
    datasets: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bad argument must fail now, not silently in a child nobody is watching."""
    artifacts = tmp_path / "artifacts"
    code = main(
        [
            "run",
            "",
            "--dataset",
            "fixture_many",
            "--artifacts",
            str(artifacts),
            "--datasets",
            str(datasets),
            "--repo-root",
            str(REPO),
        ]
    )
    assert code == 2
    assert "at least one value" in capsys.readouterr().err
    assert not artifacts.exists() or not list(artifacts.iterdir())
