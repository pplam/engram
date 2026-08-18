"""Per-run liveness for `bench ps` (§3.5.2).

`bench ps` must read progress without touching the running process, and the artifacts
alone cannot tell "crashed mid-ingest" from "still ingesting". So each run directory
carries a small state file, written atomically so a concurrent reader never sees a
half-written one.

The artifacts remain the source of truth: progress counts come from `read_ids()` on the
current stage's output. This file is a cache plus liveness, which is what keeps
"artifacts are the API" (P3) intact — no daemon, no database.

`stale` is **derived, never written**: a run whose process is gone *and* whose heartbeat
has gone cold. Requiring both means a long-running call is not mistaken for a crash, and
a just-exited process is not reported before its final write lands.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

Status = Literal["queued", "running", "done", "failed"]
Displayed = Literal["queued", "running", "done", "failed", "stale"]

# A heartbeat older than this, with no live process, reads as a crash.
STALE_AFTER_S = 120

STATE_FILE = "state.json"


class RunState(BaseModel):
    """One run's status as `bench ps` sees it. `stale` is not storable by design."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    system: str
    dataset: str
    suite: str
    status: Status
    stage: str | None = None
    # Where `stage` sits in this run's pipeline, 1-based. Optional because `queued` has no
    # stage yet, and because a state file written before these fields existed must stay
    # readable — `read_state` degrades to None on a validation error, which would blank a
    # live run out of the table.
    stage_index: int | None = None
    stage_total: int | None = None
    pid: int
    # An optional human label shown by `bench ps`. Metadata only: `run_id` is the
    # namespace, so a name need not be unique and nothing keys off it.
    name: str | None = None
    started_at_ms: int
    heartbeat_at_ms: int
    stage_progress: dict[str, int] | None = None
    # What `stage_progress` counted — "chunks", "answers". A bare 412/1382 does not say.
    progress_unit: str | None = None
    error: str | None = None


def write_state(run_dir: Path, state: RunState) -> None:
    """Write `state.json` atomically: temp file then an atomic rename."""
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / STATE_FILE
    temp = run_dir / f".{STATE_FILE}.tmp"
    temp.write_text(json.dumps(state.model_dump(mode="json"), sort_keys=True, indent=2) + "\n")
    # Path.replace is atomic within a filesystem, so a reader sees either the old file or
    # the new one and never a partial write.
    temp.replace(target)


def read_state(run_dir: Path) -> RunState | None:
    """Return the recorded state, or None if it is absent or unreadable.

    Unreadable degrades to None rather than raising: one hand-corrupted file must not
    take down a `bench ps` that is reporting on every other run.
    """
    path = run_dir / STATE_FILE
    if not path.is_file():
        return None
    try:
        raw: Any = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    try:
        return RunState.model_validate(raw)
    except ValidationError:
        return None


def is_pid_alive(pid: int) -> bool:
    """True when a process with `pid` exists and could be signalled."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # It exists; it just is not ours to signal.
        return True
    return True


def resolve_status(state: RunState, now_ms: int | None = None) -> Displayed:
    """Return the displayed status, deriving `stale` from a dead pid and a cold heartbeat."""
    if state.status != "running":
        return state.status

    now = int(time.time() * 1000) if now_ms is None else now_ms
    cold = (now - state.heartbeat_at_ms) > STALE_AFTER_S * 1000
    if cold and not is_pid_alive(state.pid):
        return "stale"
    return "running"
