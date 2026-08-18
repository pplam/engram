"""The per-run log is a file beside the artifacts, and one process writes one run's log."""

import logging
from pathlib import Path

from orchestrator.runlog import RUN_LOG, configure, get_logger


def test_configure_writes_records_to_run_log(tmp_path: Path) -> None:
    configure(tmp_path)
    get_logger("ingest").info("ingest 1/2 latency_ms=5")
    assert "ingest 1/2 latency_ms=5" in (tmp_path / RUN_LOG).read_text()


def test_configure_replaces_the_previous_run_log(tmp_path: Path) -> None:
    """One process runs one run, so a second configure must not keep writing to the first."""
    first, second = tmp_path / "a", tmp_path / "b"
    configure(first)
    configure(second)
    get_logger("plan").info("only the second run")

    assert "only the second run" not in (first / RUN_LOG).read_text()
    assert "only the second run" in (second / RUN_LOG).read_text()


def test_configure_leaves_the_root_logger_alone(tmp_path: Path) -> None:
    """A benchmark must not hijack logging for whatever process imported it."""
    configure(tmp_path)
    assert logging.getLogger().handlers == logging.getLogger().handlers
    logging.getLogger("some.other.library").info("not ours")
    assert "not ours" not in (tmp_path / RUN_LOG).read_text()


def test_each_record_carries_a_timestamp_and_a_source(tmp_path: Path) -> None:
    configure(tmp_path)
    get_logger("retrieve").info("retrieve 3/3")
    line = (tmp_path / RUN_LOG).read_text().strip()
    assert line.startswith("20"), line
    assert "retrieve" in line
    assert "INFO" in line
