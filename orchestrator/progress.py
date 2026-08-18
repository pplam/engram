"""Recording a run's progress into `state.json` (§3.5.2).

Thin on purpose: the artifacts stay the source of truth, and this only keeps the
liveness cache that `bench ps` reads. It lives beside the run rather than inside the
stages so no stage has to know that anyone is watching.

Two things make that cache legible on a run measured in hours. A stage is recorded with
its **position** in the pipeline, because "ingest" alone does not say whether a run is
nearly done. And in-stage counts are refreshed by a **background ticker** rather than
only at stage boundaries: a count written after `run_ingest` returns is a count that
never moved while ingest was actually running.

The pipeline length is passed in rather than hardcoded, since the in-process baselines
skip `ingest` and genuinely run six stages, not seven.
"""

import asyncio
import contextlib
import os
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Self

from orchestrator.artifacts import ArtifactError, JsonlArtifact
from orchestrator.runstate import RunState, Status, write_state

# How often the ticker refreshes a running stage's count. Well under `STALE_AFTER_S`, so a
# stage doing slow work still reads as alive, and cheap: one line count of a local file.
TICK_INTERVAL_S = 5.0


class Progress:
    """Writes one run's `state.json` as it moves through the stages."""

    def __init__(
        self,
        run_dir: Path,
        run_id: str,
        system: str,
        dataset: str,
        suite: str,
        stages: Sequence[str],
        name: str | None = None,
    ) -> None:
        self._run_dir = run_dir
        self._run_id = run_id
        self._system = system
        self._dataset = dataset
        self._suite = suite
        self._stages = tuple(stages)
        self._name = name
        self._started_ms = _now_ms()
        self._stage: str | None = None
        self._unit: str | None = None
        self._progress: dict[str, int] | None = None

    def __enter__(self) -> Self:
        # `queued` is a status and not a stage, so it carries no position in the pipeline
        # and `stage` stays None until the first real stage is entered.
        self._write("queued")
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object
    ) -> None:
        if exc is None:
            self._write("done")
        else:
            # Record the exception type and message, never a payload: a state file is
            # not a place for prompts or retrieved content.
            self._write("failed", error=f"{type(exc).__name__}: {exc}")

    def enter(self, stage: str, status: Status = "running") -> None:
        """Mark `stage` as the one now in progress, clearing the previous stage's count."""
        if stage not in self._stages:
            raise ValueError(f"{stage!r} is not a stage of this run: {', '.join(self._stages)}")
        self._stage = stage
        self._unit = None
        self._progress = None
        self._write(status)

    def beat(self, artifact: str, total: int, unit: str | None = None) -> None:
        """Refresh the heartbeat, counting completed IDs from the stage's own artifact."""
        self._unit = unit or self._unit
        self._write("running", progress=self._count(artifact, total))

    @contextlib.asynccontextmanager
    async def ticking(
        self,
        artifact: str,
        total: int,
        unit: str | None = None,
        interval: float = TICK_INTERVAL_S,
    ) -> AsyncIterator[None]:
        """Refresh the count every `interval` seconds while the wrapped stage runs.

        The stage stays unaware of it: this counts the artifact the stage is appending to,
        so nothing has to be threaded through a stage signature or a worker callback.
        """
        self._unit = unit or self._unit

        async def tick() -> None:
            while True:
                await asyncio.sleep(interval)
                self._write("running", progress=self._count(artifact, total))

        ticker = asyncio.create_task(tick())
        try:
            yield
        finally:
            # Cancelled before the final write, so a ticker can never outlive its stage and
            # stamp a heartbeat onto the next one. Awaited so the task is truly gone.
            ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ticker
            self._write("running", progress=self._count(artifact, total))

    def _count(self, artifact: str, total: int) -> dict[str, int] | None:
        """Return the stage's completed/total, keeping the last good count if unreadable."""
        path = self._run_dir / artifact
        if not path.is_file():
            self._progress = {"done": 0, "total": total}
            return self._progress
        # The ticker reads a file another coroutine is appending to, so it can catch a torn
        # final line. Progress is a display cache: keep the last good number rather than
        # failing a stage over the thing watching it.
        with contextlib.suppress(ArtifactError, OSError):
            self._progress = {"done": JsonlArtifact(path).count(), "total": total}
        return self._progress

    def _write(
        self,
        status: Status,
        progress: dict[str, int] | None = None,
        error: str | None = None,
    ) -> None:
        stage_index = self._stages.index(self._stage) + 1 if self._stage in self._stages else None
        write_state(
            self._run_dir,
            RunState(
                run_id=self._run_id,
                system=self._system,
                dataset=self._dataset,
                suite=self._suite,
                name=self._name,
                status=status,
                stage=self._stage,
                stage_index=stage_index,
                stage_total=len(self._stages) if stage_index else None,
                pid=os.getpid(),
                started_at_ms=self._started_ms,
                heartbeat_at_ms=_now_ms(),
                stage_progress=progress,
                progress_unit=self._unit if progress else None,
                error=error,
            ),
        )


def _now_ms() -> int:
    return int(time.time() * 1000)
