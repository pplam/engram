"""`bench ps` is a pure function of directory contents (§3.5.4)."""

import json
import os
import time
from pathlib import Path

import pytest

from orchestrator.cli import CLEAR_SCREEN, main
from orchestrator.runstate import STALE_AFTER_S, RunState, write_state


def make_run(artifacts: Path, run_id: str, **kw: object) -> Path:
    base: dict[str, object] = {
        "run_id": run_id,
        "system": "bm25",
        "dataset": "fixture_many",
        "suite": "v1",
        "status": "running",
        "stage": "ingest",
        "pid": os.getpid(),
        "started_at_ms": 1_000,
        "heartbeat_at_ms": int(__import__("time").time() * 1000),
    }
    run_dir = artifacts / run_id
    write_state(run_dir, RunState.model_validate({**base, **kw}))
    return run_dir


def test_ps_lists_a_running_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_run(tmp_path, "r1")
    assert main(["ps", "--artifacts", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "r1" in out
    assert "running" in out


def test_ps_shows_the_expected_columns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_run(tmp_path, "r1")
    main(["ps", "--artifacts", str(tmp_path)])
    header = capsys.readouterr().out.splitlines()[0]
    for column in ("RUN ID", "SYSTEM", "DATASET", "STAGE", "PROGRESS", "ELAPSED", "STATUS"):
        assert column in header


def test_stage_column_shows_the_position_in_the_pipeline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bare `ingest` does not say how much of the run is left; `2/7 ingest` does."""
    make_run(tmp_path, "r1", stage="ingest", stage_index=2, stage_total=7)
    main(["ps", "--artifacts", str(tmp_path)])
    assert "2/7 ingest" in capsys.readouterr().out


def test_stage_column_falls_back_to_the_bare_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A state file written before this column existed must still render."""
    make_run(tmp_path, "r1", stage="ingest")
    main(["ps", "--artifacts", str(tmp_path)])
    out = capsys.readouterr().out
    assert "ingest" in out
    assert "/7" not in out


def test_progress_carries_the_unit_of_the_stage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bare `412/1382` does not say what was counted; the stage's own unit does."""
    make_run(
        tmp_path,
        "r1",
        stage="answer",
        stage_progress={"done": 3, "total": 8},
        progress_unit="answers",
    )
    main(["ps", "--artifacts", str(tmp_path)])
    assert "3/8 answers" in capsys.readouterr().out


def test_json_output_carries_the_stage_position(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_run(tmp_path, "r1", stage="ingest", stage_index=2, stage_total=7)
    main(["ps", "--artifacts", str(tmp_path), "--json"])
    row = json.loads(capsys.readouterr().out)[0]
    assert (row["stage_index"], row["stage_total"]) == (2, 7)


def test_ps_hides_finished_runs_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_run(tmp_path, "running_one")
    make_run(tmp_path, "done_one", status="done", stage=None)
    out = capsys.readouterr().out
    main(["ps", "--artifacts", str(tmp_path)])
    out = capsys.readouterr().out
    assert "running_one" in out
    assert "done_one" not in out


def test_all_includes_finished_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_run(tmp_path, "done_one", status="done", stage=None)
    main(["ps", "--artifacts", str(tmp_path), "--all"])
    assert "done_one" in capsys.readouterr().out


def test_a_crashed_run_is_reported_stale(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dead pid plus a cold heartbeat is how a crash is seen without a daemon."""
    make_run(tmp_path, "crashed", pid=0, heartbeat_at_ms=0)
    main(["ps", "--artifacts", str(tmp_path)])
    assert "stale" in capsys.readouterr().out


def test_progress_is_rendered_as_done_over_total(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_run(tmp_path, "r1", stage_progress={"done": 412, "total": 1382})
    main(["ps", "--artifacts", str(tmp_path)])
    assert "412/1382" in capsys.readouterr().out


def test_json_output_is_scriptable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_run(tmp_path, "r1")
    main(["ps", "--artifacts", str(tmp_path), "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["run_id"] == "r1"
    assert rows[0]["status"] == "running"


def test_a_corrupt_state_file_does_not_break_the_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One unreadable run must not hide every other run."""
    make_run(tmp_path, "good")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "state.json").write_text("{not json")

    assert main(["ps", "--artifacts", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "good" in out
    assert "broken" in out
    assert "unknown" in out


def test_an_empty_artifacts_root_is_not_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["ps", "--artifacts", str(tmp_path)]) == 0
    assert "no runs" in capsys.readouterr().out


def test_a_missing_artifacts_root_is_not_an_error(tmp_path: Path) -> None:
    assert main(["ps", "--artifacts", str(tmp_path / "nope")]) == 0


def test_rm_deletes_a_finished_run(tmp_path: Path) -> None:
    make_run(tmp_path, "r1", status="done", stage=None)
    assert main(["rm", "r1", "--artifacts", str(tmp_path)]) == 0
    assert not (tmp_path / "r1").exists()


def test_rm_refuses_a_running_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Deleting a live run's artifacts would corrupt it mid-write."""
    make_run(tmp_path, "r1")
    assert main(["rm", "r1", "--artifacts", str(tmp_path)]) == 1
    assert (tmp_path / "r1").exists()
    assert "running" in capsys.readouterr().err


def test_rm_reports_a_missing_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["rm", "ghost", "--artifacts", str(tmp_path)]) == 2
    assert "ghost" in capsys.readouterr().err


def test_stale_elapsed_is_measured_from_the_last_heartbeat(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Elapsed on a crashed run must not keep growing after the process died."""
    make_run(
        tmp_path,
        "crashed",
        pid=0,
        started_at_ms=0,
        heartbeat_at_ms=STALE_AFTER_S * 1000,
    )
    main(["ps", "--artifacts", str(tmp_path), "--all"])
    assert "2m" in capsys.readouterr().out


def test_watch_redraws_and_stops_when_nothing_is_live(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--watch` is a clear-and-redraw loop, and it exits once every run has finished."""
    make_run(tmp_path, "r1", status="done", stage=None)
    assert main(["ps", "--artifacts", str(tmp_path), "--all", "--watch"]) == 0
    out = capsys.readouterr().out
    assert CLEAR_SCREEN in out
    assert "r1" in out


def test_watch_stops_when_interrupted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C leaves the terminal with a drawn table, not a traceback."""
    make_run(tmp_path, "r1")

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("orchestrator.cli.time.sleep", interrupt)
    assert main(["ps", "--artifacts", str(tmp_path), "--watch"]) == 0
    assert "r1" in capsys.readouterr().out


def test_watch_and_json_are_refused_together(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A redraw loop emitting repeated JSON documents is not parseable output."""
    assert main(["ps", "--artifacts", str(tmp_path), "--watch", "--json"]) == 2
    assert "--watch" in capsys.readouterr().err


def test_ps_shows_start_and_end_columns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`ELAPSED` says how long a run took but not *when*, which is what dates a row."""
    make_run(tmp_path, "r1")
    main(["ps", "--artifacts", str(tmp_path)])
    header = capsys.readouterr().out.splitlines()[0]
    assert "STARTED" in header
    assert "ENDED" in header


def test_started_renders_as_a_wall_clock_time(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Local time, because the question is "was this before or after the change I made?"."""
    started = 1_700_000_000_000
    make_run(tmp_path, "r1", started_at_ms=started)
    main(["ps", "--artifacts", str(tmp_path)])
    expected = time.strftime("%m-%d %H:%M:%S", time.localtime(started / 1000))
    assert expected in capsys.readouterr().out


def test_a_live_run_has_no_end_time(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A running run has not ended, and printing its heartbeat there would read as one."""
    make_run(tmp_path, "r1", status="running")
    row = _sole_json_row(tmp_path, capsys)
    assert row["ended_at_ms"] is None
    assert row["ended"] == "-"


def test_a_settled_run_ends_at_its_last_heartbeat(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The heartbeat is the last time the run wrote anything, so it is when it stopped.

    `RunState` records no `ended_at_ms` of its own; adding one would mean every writer
    had to set it, and a crashed run never would. The final heartbeat is what a crash
    leaves behind, which makes it the honest end for `failed` as well as `done`.
    """
    ended = 1_700_000_050_000
    make_run(tmp_path, "r1", status="done", stage=None, heartbeat_at_ms=ended)
    row = _sole_json_row(tmp_path, capsys, show_all=True)
    assert row["ended_at_ms"] == ended


def test_a_stale_run_ends_at_its_last_heartbeat(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run whose process died without writing `failed` still stopped when it went quiet."""
    ended = int(time.time() * 1000) - (STALE_AFTER_S + 60) * 1000
    # A dead pid plus a cold heartbeat is how a crash is seen without a daemon.
    make_run(tmp_path, "r1", status="running", pid=0, heartbeat_at_ms=ended)
    row = _sole_json_row(tmp_path, capsys)
    assert row["status"] == "stale"
    assert row["ended_at_ms"] == ended


def test_json_carries_epoch_millis_not_only_the_rendered_string(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A consumer sorting or diffing runs needs the number, not a localised string."""
    started = 1_700_000_000_000
    make_run(tmp_path, "r1", started_at_ms=started)
    assert _sole_json_row(tmp_path, capsys)["started_at_ms"] == started


def test_an_unreadable_run_still_reports_no_times(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The degraded row must fill both new columns rather than misalign the table."""
    (tmp_path / "corrupt").mkdir()
    (tmp_path / "corrupt" / "state.json").write_text("{not json")
    main(["ps", "--artifacts", str(tmp_path)])
    assert "corrupt" in capsys.readouterr().out


def _sole_json_row(
    artifacts: Path, capsys: pytest.CaptureFixture[str], show_all: bool = False
) -> dict[str, object]:
    argv = ["ps", "--artifacts", str(artifacts), "--json"]
    if show_all:
        argv.append("--all")
    main(argv)
    rows: list[dict[str, object]] = json.loads(capsys.readouterr().out)
    return rows[0]
