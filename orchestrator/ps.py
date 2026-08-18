"""Reading run progress off the filesystem for `bench ps` (§3.5.4).

A directory scan of `artifacts/*/state.json` is the whole mechanism — no daemon, no
database, no state in the CLI process. That keeps "artifacts are the API" (P3) intact
and means `bench ps` cannot perturb a run it is reporting on.

An unreadable run degrades to `unknown` rather than raising: one hand-corrupted state
file must not hide every other run in the table.
"""

import time
from dataclasses import dataclass
from pathlib import Path

from orchestrator.runstate import STATE_FILE, read_state, resolve_status

# Statuses that have not settled, and so show without `--all`. `unknown` is included
# deliberately: a run whose state cannot be read is exactly the one worth surfacing.
LIVE = ("queued", "running", "stale", "unknown")


@dataclass(frozen=True)
class PsRow:
    """One line of `bench ps`, already resolved for display."""

    run_id: str
    system: str
    dataset: str
    stage: str
    progress: str
    elapsed_s: int
    status: str
    name: str = "-"
    # The pipeline position behind the rendered `stage`, kept as numbers so `--json`
    # consumers do not have to parse "2/7 ingest" back apart.
    stage_index: int | None = None
    stage_total: int | None = None
    # When the run began and stopped. Rendered for the table, and kept as epoch millis
    # beside it so a `--json` consumer can sort or diff without parsing a localised
    # string. `ended_*` is None while a run is still live, which is not the same as a
    # run that ended at its start time.
    started: str = "-"
    ended: str = "-"
    started_at_ms: int | None = None
    ended_at_ms: int | None = None

    @property
    def is_live(self) -> bool:
        """True for runs that have not settled, which is the default `ps` filter."""
        return self.status in LIVE


def _elapsed_s(started_at_ms: int, until_ms: int) -> int:
    return max((until_ms - started_at_ms) // 1000, 0)


# Live statuses keep accruing time, so they have a start but no end.
_RUNNING = ("running", "queued")


def format_clock(at_ms: int | None) -> str:
    """Render an epoch-millis instant as local `MM-DD HH:MM:SS`, or `-` for None.

    Local time because the question this column answers is "did this run happen before or
    after the change I made?", which is asked in the reader's own timezone. The year is
    dropped and the date kept: a run from last week is worth distinguishing from one an
    hour ago, and one from last year is not something this table has to disambiguate.
    """
    if at_ms is None:
        return "-"
    return time.strftime("%m-%d %H:%M:%S", time.localtime(at_ms / 1000))


def _row(run_dir: Path, now_ms: int) -> PsRow:
    state = read_state(run_dir)
    if state is None:
        # The directory exists but its state is missing or unreadable. Say so plainly
        # rather than guessing, and keep the run visible.
        return PsRow(
            run_id=run_dir.name,
            system="-",
            dataset="-",
            stage="-",
            progress="-",
            elapsed_s=0,
            status="unknown",
        )

    status = resolve_status(state, now_ms=now_ms)
    progress = "-"
    if state.stage_progress:
        done = state.stage_progress.get("done", 0)
        total = state.stage_progress.get("total", 0)
        # The unit is what makes the pair readable: 412/1382 chunks, not a bare ratio.
        progress = f"{done}/{total}"
        if state.progress_unit:
            progress = f"{progress} {state.progress_unit}"

    # A settled or crashed run stopped accruing time at its last heartbeat; only a live
    # run is still running the clock. That same heartbeat is the run's end time: nothing
    # writes an `ended_at_ms`, and a run killed mid-stage never would, so the last moment
    # it wrote anything is the honest answer for `failed` and `stale` as much as `done`.
    live = status in _RUNNING
    until = now_ms if live else state.heartbeat_at_ms
    ended_at_ms = None if live else state.heartbeat_at_ms

    return PsRow(
        run_id=state.run_id,
        system=state.system,
        dataset=state.dataset,
        stage=format_stage(state.stage, state.stage_index, state.stage_total),
        progress=progress,
        elapsed_s=_elapsed_s(state.started_at_ms, until),
        status=status,
        name=state.name or "-",
        stage_index=state.stage_index,
        stage_total=state.stage_total,
        started=format_clock(state.started_at_ms),
        ended=format_clock(ended_at_ms),
        started_at_ms=state.started_at_ms,
        ended_at_ms=ended_at_ms,
    )


def format_stage(stage: str | None, index: int | None, total: int | None) -> str:
    """Render the stage with its place in the pipeline: `2/7 ingest`.

    The position leads so the column sorts and scans by how far along a run is, which is
    the question the table is usually being asked. A state file without the counts — one
    written before they existed — renders the bare name rather than dropping the row.
    """
    if stage is None:
        return "-"
    if index is None or total is None:
        return stage
    return f"{index}/{total} {stage}"


def collect_rows(artifacts: Path, now_ms: int | None = None) -> list[PsRow]:
    """Return one row per run directory under `artifacts`, sorted by run id."""
    if not artifacts.is_dir():
        return []
    now = int(time.time() * 1000) if now_ms is None else now_ms
    directories = [
        path for path in sorted(artifacts.iterdir()) if path.is_dir() and _looks_like_run(path)
    ]
    return [_row(path, now) for path in directories]


def _looks_like_run(path: Path) -> bool:
    """True for a directory that holds run artifacts rather than something unrelated."""
    return (path / STATE_FILE).exists() or (path / "manifest.json").exists()


def format_elapsed(seconds: int) -> str:
    """Render a duration the way `docker ps` does: compact, no units spelled out."""
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h{mins:02d}m"


def render_table(rows: list[PsRow]) -> str:
    """Render rows as an aligned plain-text table (stdlib only, no `rich`)."""
    if not rows:
        return "no runs"

    headers = (
        "RUN ID",
        "NAME",
        "SYSTEM",
        "DATASET",
        "STAGE",
        "PROGRESS",
        "STARTED",
        "ENDED",
        "ELAPSED",
        "STATUS",
    )
    body = [
        (
            row.run_id,
            row.name,
            row.system,
            row.dataset,
            row.stage,
            row.progress,
            row.started,
            row.ended,
            format_elapsed(row.elapsed_s),
            row.status,
        )
        for row in rows
    ]
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *body, strict=True)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)).rstrip()]
    lines.extend(
        "  ".join(cell.ljust(w) for cell, w in zip(entry, widths, strict=True)).rstrip()
        for entry in body
    )
    return "\n".join(lines)
