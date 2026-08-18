"""`--resume` must refuse to splice two configurations into one published row (§4.5).

Artifacts are append-only, so resuming is just "skip the IDs already present". That is
safe *only* if the run being continued is the same run: a resume against a different
system, dataset, or suite would produce one directory whose rows came from two
configurations, and one `report.json` averaging across them.
"""

import json
from pathlib import Path

import pytest

from orchestrator.run import RunError, check_resumable, resumed_config

MANIFEST = {
    "run_id": "8c1e04b7f2a9",
    "system": {"name": "bm25"},
    "suite": {"version": "v1"},
    "dataset": {"name": "locomo-refined"},
}


def bundle(tmp_path: Path, **overrides: object) -> Path:
    run_dir = tmp_path / "8c1e04b7f2a9"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({**MANIFEST, **overrides}))
    return run_dir


def test_a_matching_configuration_resumes(tmp_path: Path) -> None:
    check_resumable(bundle(tmp_path), system="bm25", dataset="locomo-refined", suite="v1")


def test_a_different_system_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RunError, match="system"):
        check_resumable(bundle(tmp_path), system="no_memory", dataset="locomo-refined", suite="v1")


def test_a_different_dataset_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RunError, match="dataset"):
        check_resumable(bundle(tmp_path), system="bm25", dataset="longmemeval-cleaned", suite="v1")


def test_a_different_suite_is_refused(tmp_path: Path) -> None:
    """A suite bump changes what a row means, so it cannot share a directory."""
    with pytest.raises(RunError, match="suite"):
        check_resumable(bundle(tmp_path), system="bm25", dataset="locomo-refined", suite="v2")


def test_the_message_names_both_values(tmp_path: Path) -> None:
    """ "refused" without the two values leaves the user guessing which is wrong."""
    with pytest.raises(RunError) as err:
        check_resumable(bundle(tmp_path), system="no_memory", dataset="locomo-refined", suite="v1")
    assert "bm25" in str(err.value)
    assert "no_memory" in str(err.value)


def test_a_run_with_no_manifest_is_refused(tmp_path: Path) -> None:
    (tmp_path / "8c1e04b7f2a9").mkdir()
    with pytest.raises(RunError, match="manifest"):
        check_resumable(
            tmp_path / "8c1e04b7f2a9", system="bm25", dataset="locomo-refined", suite="v1"
        )


def test_an_unreadable_manifest_is_refused(tmp_path: Path) -> None:
    run_dir = tmp_path / "8c1e04b7f2a9"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{not json")
    with pytest.raises(RunError, match="manifest"):
        check_resumable(run_dir, system="bm25", dataset="locomo-refined", suite="v1")


def test_a_different_limit_is_refused(tmp_path: Path) -> None:
    """A resume that replans at a new limit rewrites the plan the artifacts were built for."""
    with pytest.raises(RunError, match="limit"):
        check_resumable(
            bundle(tmp_path, limit=1),
            system="bm25",
            dataset="locomo-refined",
            suite="v1",
            limit=None,
        )


def test_a_matching_limit_resumes(tmp_path: Path) -> None:
    check_resumable(
        bundle(tmp_path, limit=1), system="bm25", dataset="locomo-refined", suite="v1", limit=1
    )


class TestResumedConfig:
    """`--resume <id>` alone: the manifest already names the configuration (§4.5).

    Retyping `--dataset` and the system is friction with no upside — the only thing they
    can do is match the manifest or be rejected. Reading them back also closes the gap
    where an *omitted* `--limit` silently replanned the run at a different size.
    """

    def test_it_returns_the_recorded_configuration(self, tmp_path: Path) -> None:
        config = resumed_config(bundle(tmp_path, limit=3))
        assert (config.system, config.dataset, config.suite, config.limit) == (
            "bm25",
            "locomo-refined",
            "v1",
            3,
        )

    def test_an_absent_limit_reads_as_none(self, tmp_path: Path) -> None:
        assert resumed_config(bundle(tmp_path)).limit is None

    def test_the_recorded_name_is_carried_over(self, tmp_path: Path) -> None:
        """The `bench ps` label belongs to the run, so a resume must not silently drop it."""
        assert resumed_config(bundle(tmp_path, name="nightly")).name == "nightly"

    def test_a_run_with_no_manifest_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "8c1e04b7f2a9").mkdir()
        with pytest.raises(RunError, match="manifest"):
            resumed_config(tmp_path / "8c1e04b7f2a9")

    def test_an_unreadable_manifest_is_refused(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "8c1e04b7f2a9"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text("{not json")
        with pytest.raises(RunError, match="manifest"):
            resumed_config(run_dir)

    @pytest.mark.parametrize("missing", ["system", "dataset", "suite"])
    def test_a_manifest_missing_a_field_is_refused(self, tmp_path: Path, missing: str) -> None:
        """Guessing a blank system or dataset would run the wrong configuration."""
        run_dir = tmp_path / "8c1e04b7f2a9"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(
            json.dumps({k: v for k, v in MANIFEST.items() if k != missing})
        )
        with pytest.raises(RunError, match=missing):
            resumed_config(run_dir)
