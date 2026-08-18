"""Progress records stage position and in-flight counts in `state.json` (§3.5.2)."""

import asyncio
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from orchestrator.artifacts import JsonlArtifact
from orchestrator.progress import Progress
from orchestrator.runstate import RunState, read_state

STAGES = ("plan", "ingest", "retrieve", "answer", "judge", "verify", "score")
IN_PROCESS = ("plan", "retrieve", "answer", "judge", "verify", "score")


def progress_for(run_dir: Path, stages: Sequence[str] = STAGES) -> Progress:
    return Progress(
        run_dir,
        run_id="r1",
        system="bm25",
        dataset="fixture_many",
        suite="v1",
        stages=stages,
    )


def state_of(run_dir: Path) -> RunState:
    state = read_state(run_dir)
    assert state is not None
    return state


def test_enter_records_the_stage_position(tmp_path: Path) -> None:
    progress_for(tmp_path).enter("retrieve")
    state = state_of(tmp_path)
    assert (state.stage, state.stage_index, state.stage_total) == ("retrieve", 3, 7)


def test_a_shorter_pipeline_reports_its_own_total(tmp_path: Path) -> None:
    """An in-process baseline skips `ingest`, so its retrieve is 2 of 6, not 3 of 7."""
    progress_for(tmp_path, IN_PROCESS).enter("retrieve")
    state = state_of(tmp_path)
    assert (state.stage_index, state.stage_total) == (2, 6)


def test_queued_carries_no_stage_position(tmp_path: Path) -> None:
    """`queued` is a status, not a stage: it has no place in the sequence."""
    with progress_for(tmp_path):
        state = state_of(tmp_path)
        assert state.status == "queued"
        assert state.stage_index is None


def test_an_unknown_stage_is_rejected(tmp_path: Path) -> None:
    """A typo must fail here, not silently show a run stuck at an unnumbered stage."""
    with pytest.raises(ValueError, match="rerank"):
        progress_for(tmp_path).enter("rerank")


async def _wait_for_done(run_dir: Path, done: int, give_up_after: float = 5.0) -> RunState:
    """Poll `state.json` until it reports `done` completed items."""
    deadline = time.monotonic() + give_up_after
    while time.monotonic() < deadline:
        state = read_state(run_dir)
        if state and state.stage_progress and state.stage_progress["done"] == done:
            return state
        await asyncio.sleep(0.01)
    raise AssertionError(f"state.json never reported done={done}: {read_state(run_dir)}")


async def test_ticking_reports_progress_while_the_stage_is_still_running(tmp_path: Path) -> None:
    """The count must move mid-stage; a stage-boundary-only write shows a frozen table."""
    progress = progress_for(tmp_path)
    progress.enter("ingest")
    out = JsonlArtifact(tmp_path / "ingest.jsonl")

    async with progress.ticking("ingest.jsonl", total=3, interval=0.01):
        out.append({"id": "a"})
        state = await _wait_for_done(tmp_path, 1)
        assert state.stage_progress == {"done": 1, "total": 3}
        assert state.status == "running"
        out.append({"id": "b"})
        await _wait_for_done(tmp_path, 2)


async def test_ticking_writes_the_final_count_on_exit(tmp_path: Path) -> None:
    progress = progress_for(tmp_path)
    progress.enter("ingest")
    out = JsonlArtifact(tmp_path / "ingest.jsonl")

    async with progress.ticking("ingest.jsonl", total=2, interval=10.0):
        out.append({"id": "a"})
        out.append({"id": "b"})

    assert state_of(tmp_path).stage_progress == {"done": 2, "total": 2}


async def test_ticking_stops_writing_once_the_stage_is_over(tmp_path: Path) -> None:
    """A leaked ticker would keep stamping heartbeats onto a stage that has moved on."""
    progress = progress_for(tmp_path)
    progress.enter("ingest")
    async with progress.ticking("ingest.jsonl", total=1, interval=0.01):
        await asyncio.sleep(0.05)

    settled = state_of(tmp_path).heartbeat_at_ms
    await asyncio.sleep(0.1)
    assert state_of(tmp_path).heartbeat_at_ms == settled


async def test_ticking_lets_the_stage_failure_through(tmp_path: Path) -> None:
    """Progress is a liveness cache; it must never swallow or replace a stage error."""
    progress = progress_for(tmp_path)
    progress.enter("ingest")

    with pytest.raises(RuntimeError, match="contract violated"):
        async with progress.ticking("ingest.jsonl", total=1, interval=0.01):
            raise RuntimeError("contract violated")


async def test_a_partly_written_record_does_not_fail_the_run(tmp_path: Path) -> None:
    """The ticker reads a file being appended to, so a torn final line must not raise."""
    progress = progress_for(tmp_path)
    progress.enter("ingest")
    (tmp_path / "ingest.jsonl").write_text('{"id": "a"}\n{"id": "b"')

    async with progress.ticking("ingest.jsonl", total=2, interval=0.01):
        await asyncio.sleep(0.05)

    # Unreadable counts leave the last good number in place rather than taking down ingest.
    assert state_of(tmp_path).status == "running"
