"""`bench logs` prints the run's log, not just its status line.

Detaching makes this the only way to see what a run is doing, so a `logs` command that
shows four summary fields and not the log itself is the wrong command.
"""

from pathlib import Path

import pytest

from orchestrator.cli import main
from orchestrator.runlog import RUN_LOG
from orchestrator.runstate import RunState, write_state


def make_run(artifacts: Path, run_id: str = "abc123def456", log: str | None = None) -> Path:
    run_dir = artifacts / run_id
    write_state(
        run_dir,
        RunState(
            run_id=run_id,
            system="bm25",
            dataset="fixture_many",
            suite="v1",
            status="running",
            stage="ingest",
            pid=1,
            started_at_ms=1_000,
            heartbeat_at_ms=2_000,
        ),
    )
    if log is not None:
        (run_dir / RUN_LOG).write_text(log)
    return run_dir


def test_the_status_header_shows_the_stage_position(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same position `bench ps` shows, so the two views do not disagree."""
    artifacts = tmp_path / "artifacts"
    run_dir = make_run(artifacts)
    state = RunState.model_validate(
        {
            "run_id": "abc123def456",
            "system": "bm25",
            "dataset": "fixture_many",
            "suite": "v1",
            "status": "running",
            "stage": "ingest",
            "stage_index": 2,
            "stage_total": 7,
            "pid": 1,
            "started_at_ms": 1_000,
            "heartbeat_at_ms": 2_000,
        }
    )
    write_state(run_dir, state)
    main(["logs", "abc123def456", "--artifacts", str(artifacts)])
    assert "2/7 ingest" in capsys.readouterr().out


def test_logs_prints_the_run_log(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifacts = tmp_path / "artifacts"
    make_run(artifacts, log="ingest start chunks=2\ningest done written=2\n")
    assert main(["logs", "abc123def456", "--artifacts", str(artifacts)]) == 0
    out = capsys.readouterr().out
    assert "ingest start chunks=2" in out
    assert "ingest done written=2" in out


def test_logs_still_reports_the_status_header(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The header answers "is it alive"; the log answers "what is it doing"."""
    artifacts = tmp_path / "artifacts"
    make_run(artifacts, log="ingest start\n")
    main(["logs", "abc123def456", "--artifacts", str(artifacts)])
    out = capsys.readouterr().out
    assert "bm25" in out
    assert "ingest" in out


def test_logs_says_so_when_there_is_no_log_yet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A just-launched run has a state file before it has a log; that is not an error."""
    artifacts = tmp_path / "artifacts"
    make_run(artifacts, log=None)
    assert main(["logs", "abc123def456", "--artifacts", str(artifacts)]) == 0
    assert "no log" in capsys.readouterr().out.lower()


def test_logs_tail_limits_how_much_it_prints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An ingest of 2000 chunks writes 2000 lines; a default dump of all of them is unusable."""
    artifacts = tmp_path / "artifacts"
    make_run(artifacts, log="".join(f"line {n}\n" for n in range(1, 101)))
    assert main(["logs", "abc123def456", "--artifacts", str(artifacts), "--tail", "3"]) == 0
    out = capsys.readouterr().out
    assert "line 100" in out
    assert "line 97" not in out


def test_logs_reports_a_recorded_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "abc123def456"
    write_state(
        run_dir,
        RunState(
            run_id="abc123def456",
            system="bm25",
            dataset="fixture_many",
            suite="v1",
            status="failed",
            stage="answer",
            pid=1,
            started_at_ms=1_000,
            heartbeat_at_ms=2_000,
            error="ModelError: model call returned HTTP 400",
        ),
    )
    (run_dir / RUN_LOG).write_text("answer start\n")
    main(["logs", "abc123def456", "--artifacts", str(artifacts)])
    assert "HTTP 400" in capsys.readouterr().out
