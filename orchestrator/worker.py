"""One run per subprocess (§3.5.3).

A batch could run its jobs as tasks in one event loop, and that would be less code. It
would also be wrong on two counts:

- A segfault or an OOM kill in one run would take the whole batch down with it. Runs are
  long and expensive, so losing the siblings of a crashed run is a real cost.
- `pid` in `state.json` only means something if runs are separate processes. `bench stop`
  signals that pid, and `stale` is derived from it being gone. Both are meaningless when
  every run shares the parent's pid.

So the parent expands the cross product, bounds concurrency, and shells out to
`python -m orchestrator.cli run` once per job. The child is the same code path a user
invokes by hand, which keeps one way of running a run rather than two.
"""

import asyncio
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from orchestrator.scheduler import Job


@dataclass(frozen=True)
class ChildResult:
    """What one child process exited with."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        """True when the child completed its run successfully."""
        return self.returncode == 0


def child_argv(
    job: Job,
    suite: str,
    artifacts: Path,
    datasets: Path,
    repo_root: Path,
    registry: Path,
    limit: int | None,
    name: str | None = None,
) -> list[str]:
    """Return the command that runs exactly one job in a child process.

    Always `--foreground`: this child *is* the run. A batch parent awaits it and reads its
    stdout for the report row, and a detached launcher has already done the detaching — in
    either case a child that spawned a grandchild and exited would break its caller.
    """
    argv = [
        sys.executable,
        "-m",
        "orchestrator.cli",
        "run",
        job.system,
        "--foreground",
        "--dataset",
        job.dataset,
        "--suite",
        suite,
        # The parent generated the id so it can label this child's output and track it;
        # a user never passes this. Same shape as --contended below.
        "--run-id",
        job.run_id,
        "--artifacts",
        str(artifacts),
        "--datasets",
        str(datasets),
        "--repo-root",
        str(repo_root),
        "--registry",
        str(registry),
    ]
    if limit is not None:
        argv += ["--limit", str(limit)]
    # Nothing is passed about the model endpoint: it is the gateway for every run, and the
    # credential is read from the environment the child inherits, so no key or URL ever
    # reaches an argument list.
    if name:
        argv += ["--name", name]
    # Contention is a property of the batch, not of the run: a child on its own cannot
    # tell that a sibling holds the same system, so the parent has to say so.
    if job.contended:
        argv.append("--contended")
    return argv


def spawn_detached(argv: list[str], cwd: Path) -> int:
    """Start `argv` fully detached and return its pid, without waiting for it.

    Detached means three things, and all three matter for a run measured in hours:

    - **Its own session** (`start_new_session`), so it is not in the launching shell's
      process group. A Ctrl-C in that terminal signals the group, and a run that dies on
      the keystroke that was meant to stop *watching* it is the bug this fixes.
    - **No inherited stdio.** The child writes `run.log` and `state.json`; its streams go
      to `os.devnull` so it cannot block on a pipe nobody drains, and so closing the
      terminal cannot hand it a SIGHUP-on-write.
    - **No wait.** The parent returns as soon as the child exists. `bench ps` and
      `bench logs` are how the run is observed from then on.

    `bench stop` still works: the child records its own pid in `state.json`, and a new
    session does not make it unsignalable.
    """
    with Path(os.devnull).open("rb") as sink_in, Path(os.devnull).open("ab") as sink_out:
        child = subprocess.Popen(  # noqa: S603 - argv is built by `child_argv`, never a shell
            argv,
            cwd=cwd,
            stdin=sink_in,
            stdout=sink_out,
            stderr=sink_out,
            start_new_session=True,
        )
    return child.pid


async def spawn(argv: list[str], cwd: Path) -> ChildResult:
    """Run `argv` to completion in a child process and return its outcome.

    Output is captured rather than inherited so that concurrent children cannot interleave
    half-lines into the parent's table.
    """
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await process.communicate()
    except asyncio.CancelledError:
        # SIGINT has to drain: ask the child to stop, let it finish its final write, and
        # let the cancellation carry on up. The run stays resumable by ID.
        process.send_signal(2)
        await process.wait()
        raise
    return ChildResult(
        returncode=process.returncode if process.returncode is not None else -1,
        stdout=out.decode(errors="replace"),
        stderr=err.decode(errors="replace"),
    )


def describe_failure(result: ChildResult) -> str:
    """Return a one-line reason a child failed, naming a signal when it was killed."""
    if result.returncode < 0:
        return f"run died on signal {-result.returncode}"
    detail = result.stderr.strip().splitlines()
    tail = detail[-1] if detail else "no error output"
    return f"exit {result.returncode}: {tail}"
