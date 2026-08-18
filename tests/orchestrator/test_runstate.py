"""`state.json` is a liveness cache; the artifacts stay the source of truth (§3.5.2)."""

import json
import os
from pathlib import Path

import pytest

from orchestrator.runstate import (
    STALE_AFTER_S,
    RunState,
    is_pid_alive,
    read_state,
    resolve_status,
    write_state,
)


def state(**kw: object) -> RunState:
    base: dict[str, object] = {
        "run_id": "r1",
        "system": "bm25",
        "dataset": "fixture_many",
        "suite": "v1",
        "status": "running",
        "stage": "ingest",
        "pid": os.getpid(),
        "started_at_ms": 1_000,
        "heartbeat_at_ms": 2_000,
    }
    return RunState.model_validate({**base, **kw})


def test_state_round_trips_through_the_file(tmp_path: Path) -> None:
    write_state(tmp_path, state())
    assert read_state(tmp_path) == state()


def test_the_write_is_atomic(tmp_path: Path) -> None:
    """`bench ps` reads while a run writes; a half-written file must never be visible."""
    write_state(tmp_path, state())
    write_state(tmp_path, state(stage="retrieve"))
    assert read_state(tmp_path) is not None
    assert not list(tmp_path.glob("*.tmp"))


def test_a_missing_state_file_reads_as_none(tmp_path: Path) -> None:
    assert read_state(tmp_path) is None


def test_a_corrupt_state_file_degrades_instead_of_crashing(tmp_path: Path) -> None:
    """A hand-corrupted state file must not take down `bench ps`."""
    (tmp_path / "state.json").write_text("{not json")
    assert read_state(tmp_path) is None


def test_a_state_file_missing_fields_degrades_instead_of_crashing(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text(json.dumps({"run_id": "r1"}))
    assert read_state(tmp_path) is None


def test_progress_is_recorded_for_the_current_stage(tmp_path: Path) -> None:
    write_state(tmp_path, state(stage_progress={"done": 412, "total": 1382}))
    recorded = read_state(tmp_path)
    assert recorded is not None
    assert recorded.stage_progress == {"done": 412, "total": 1382}


def test_a_live_process_is_detected() -> None:
    assert is_pid_alive(os.getpid())


def test_a_pid_that_is_gone_is_detected() -> None:
    """PID 0 is never a real user process on the platforms this runs on."""
    assert not is_pid_alive(0)


def test_a_running_state_with_a_live_pid_stays_running() -> None:
    assert resolve_status(state(), now_ms=2_500) == "running"


def test_a_running_state_whose_process_died_is_stale() -> None:
    """`stale` is derived, never written: it is how crashes are seen without a daemon."""
    dead = state(pid=0, heartbeat_at_ms=0)
    assert resolve_status(dead, now_ms=STALE_AFTER_S * 1000 + 1) == "stale"


def test_a_cold_heartbeat_alone_is_not_stale_while_the_process_lives() -> None:
    """A long ingest call can outlast the threshold; a live pid is the stronger signal."""
    cold = state(heartbeat_at_ms=0)
    assert resolve_status(cold, now_ms=STALE_AFTER_S * 1000 + 1) == "running"


def test_a_dead_pid_with_a_warm_heartbeat_is_not_yet_stale() -> None:
    """Both conditions are required, so a just-exited run is not misreported."""
    assert resolve_status(state(pid=0), now_ms=2_100) == "running"


@pytest.mark.parametrize("status", ["done", "failed", "queued"])
def test_a_finished_status_is_never_reinterpreted(status: str) -> None:
    """Only a `running` state can be stale; a recorded outcome is final."""
    settled = state(status=status, pid=0, heartbeat_at_ms=0)
    assert resolve_status(settled, now_ms=STALE_AFTER_S * 1000 + 1) == status


def test_an_error_is_recorded_on_failure(tmp_path: Path) -> None:
    write_state(tmp_path, state(status="failed", error="upstream returned HTTP 500"))
    recorded = read_state(tmp_path)
    assert recorded is not None
    assert recorded.error == "upstream returned HTTP 500"
