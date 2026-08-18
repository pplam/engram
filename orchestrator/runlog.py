"""One log file per run, beside that run's artifacts.

A run is long and now detaches by default, so its output cannot be the terminal it was
launched from. `artifacts/<run_id>/run.log` is where a stage says what it is doing, and
`bench logs <id>` reads it back.

Scoped to the `engram` logger rather than the root one: importing the harness must not
reconfigure logging for whatever process did the importing. One process runs exactly one
run, so `configure` replaces its handler rather than accumulating them.

The discipline from ARCH applies here without exception: **IDs, statuses, counts, and
latencies only.** No prompts, no messages, no retrieved content, no keys.
"""

import logging
from pathlib import Path

RUN_LOG = "run.log"

# Every logger a stage asks for hangs off this one, so one handler catches them all and
# nothing else in the process is touched.
ROOT = "engram"

_FORMAT = "%(asctime)s %(levelname)-5s %(name)s %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def configure(run_dir: Path, level: int = logging.INFO) -> None:
    """Point the `engram` logger at `run_dir/run.log`, replacing any earlier destination."""
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(ROOT)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(run_dir / RUN_LOG, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    # A benchmark's log is its own. Without this, records also travel to whatever the
    # embedding process configured on the root logger.
    logger.propagate = False


def get_logger(stage: str) -> logging.Logger:
    """Return the logger one stage writes to."""
    return logging.getLogger(f"{ROOT}.{stage}")
