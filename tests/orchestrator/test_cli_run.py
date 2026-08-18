"""`bench run` drives a run end to end; `bench rescore` replays `score` alone (§2.5, §2.10).

Run ids are generated, not passed in, so nothing here can hardcode one. A run is located
the way a user locates it: by the id `run` printed, or by reading the manifest that says
what the run *is*. That is the point of the change — an id is an opaque handle.
"""

import os
import re
import shutil
from pathlib import Path

import pytest

from orchestrator.artifacts import read_json
from orchestrator.cli import main
from orchestrator.suite import CURRENT_SUITE

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "datasets" / "fixtures"


@pytest.fixture
def datasets(tmp_path: Path) -> Path:
    dest = tmp_path / "datasets"
    shutil.copytree(FIXTURES, dest)
    return dest


def run_argv(datasets: Path, artifacts: Path, system: str = "bm25", *extra: str) -> list[str]:
    """One run, attached, so the assertions below can read what it produced.

    `bench run` detaches by default; these tests are about what a run *does*, and a
    detached launcher returns before there is anything to assert on. Detaching itself is
    covered in `test_cli_detach.py`.
    """
    return [
        "run",
        system,
        "--foreground",
        "--dataset",
        "fixture_many",
        "--suite",
        "v1",
        "--artifacts",
        str(artifacts),
        "--datasets",
        str(datasets),
        "--repo-root",
        str(REPO),
        *extra,
    ]


def batch_argv(datasets: Path, artifacts: Path, systems: str, sets: str, *extra: str) -> list[str]:
    """A blocking batch that makes no model call, so it can be asserted on offline.

    Two flags carry that, and both are load-bearing:

    `--foreground`, because a cross product detaches by default now and a detached launcher
    returns before any artifact exists. Detaching itself is covered in
    `test_cli_batch_detach.py`.

    `--limit 0`, because a batch child is a *real subprocess* in a fresh interpreter
    (`_run_batch` spawns one per job), which `conftest`'s offline transport cannot patch —
    it patches this process only. With questions planned, those children reach the real
    gateway and spend real money. Planning none leaves `answer` and `judge` nothing to
    call, while `plan`, `ingest`, `retrieve`, `score`, and `verify` — everything these
    tests assert on — still run and still write `report.json`.
    """
    return [
        "run",
        systems,
        "--dataset",
        sets,
        "--foreground",
        "--limit",
        "0",
        "--suite",
        "v1",
        "--artifacts",
        str(artifacts),
        "--datasets",
        str(datasets),
        "--repo-root",
        str(REPO),
        *extra,
    ]


def find_run(artifacts: Path, system: str, dataset: str = "fixture_many") -> Path:
    """Return the one run directory whose manifest names this system and dataset."""
    for path in sorted(artifacts.iterdir()):
        manifest = path / "manifest.json"
        if not manifest.is_file():
            continue
        recorded = read_json(manifest)
        if recorded["system"]["name"] == system and recorded["dataset"]["name"] == dataset:
            return path
    raise AssertionError(f"no run for {system}/{dataset} under {artifacts}")


def sole_run(artifacts: Path) -> Path:
    """Return the only run directory under `artifacts`."""
    runs = [p for p in sorted(artifacts.iterdir()) if (p / "manifest.json").is_file()]
    assert len(runs) == 1, runs
    return runs[0]


def test_run_writes_a_report(datasets: Path, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    assert main(run_argv(datasets, artifacts)) == 0
    assert (sole_run(artifacts) / "report.json").is_file()


def test_run_prints_its_generated_id_before_the_row(
    datasets: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The id comes first so an interrupted run is resumable by an id already seen."""
    artifacts = tmp_path / "artifacts"
    main(run_argv(datasets, artifacts))
    first = capsys.readouterr().out.splitlines()[0].strip()
    assert re.fullmatch(r"[0-9a-f]{12}", first)
    assert (artifacts / first / "report.json").is_file()


def test_run_prints_the_row(
    datasets: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(run_argv(datasets, tmp_path / "artifacts"))
    out = capsys.readouterr().out
    assert "bm25" in out
    assert "fixture_many" in out


def test_run_accepts_an_in_process_baseline_without_a_registry(
    datasets: Path, tmp_path: Path
) -> None:
    """oracle_gold has no adapter, so it must not need a registry entry."""
    artifacts = tmp_path / "artifacts"
    assert main(run_argv(datasets, artifacts, "oracle_gold")) == 0
    report = read_json(sole_run(artifacts) / "report.json")
    assert report["row"]["in_process"] is True


def test_run_honours_limit(datasets: Path, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    main(run_argv(datasets, artifacts, "bm25", "--limit", "2"))
    report = read_json(sole_run(artifacts) / "report.json")
    assert report["row"]["completeness"]["planned_questions"] == 2


def test_run_records_a_name_without_making_it_an_identifier(datasets: Path, tmp_path: Path) -> None:
    """`--name` is a label for `bench ps`; the run id is still the namespace."""
    artifacts = tmp_path / "artifacts"
    assert main(run_argv(datasets, artifacts, "bm25", "--name", "tuning sweep")) == 0
    run_dir = sole_run(artifacts)
    assert read_json(run_dir / "manifest.json")["name"] == "tuning sweep"
    assert run_dir.name != "tuning sweep"


def test_the_default_suite_is_the_current_one(datasets: Path, tmp_path: Path) -> None:
    """Defaulting to a legacy suite would pin no providers — the D11 bug, by default.

    Spelled out rather than built by the helpers above, because both of them pass
    `--suite v1` and this is the one test that must leave the flag off. One system and one
    dataset, so it stays in this process and `conftest`'s offline transport still applies.
    """
    artifacts = tmp_path / "artifacts"
    argv = [
        "run",
        "bm25",
        "--dataset",
        "fixture_many",
        "--foreground",
        "--artifacts",
        str(artifacts),
        "--datasets",
        str(datasets),
        "--repo-root",
        str(REPO),
    ]
    assert main(argv) == 0
    manifest = read_json(find_run(artifacts, "bm25") / "manifest.json")
    assert manifest["suite"]["version"] == CURRENT_SUITE
    assert manifest["models"]["providers"]["chat"]


def test_run_reports_an_unknown_system_without_a_traceback(
    datasets: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(run_argv(datasets, tmp_path / "artifacts", "nope")) == 2
    assert "nope" in capsys.readouterr().err


def test_rescore_recomputes_the_report_from_artifacts(datasets: Path, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    main(run_argv(datasets, artifacts))
    run_id = sole_run(artifacts).name
    before = (artifacts / run_id / "report.json").read_bytes()

    (artifacts / run_id / "report.json").unlink()
    assert main(["rescore", run_id, "--artifacts", str(artifacts)]) == 0
    assert (artifacts / run_id / "report.json").read_bytes() == before


def test_rescore_accepts_an_unambiguous_id_prefix(datasets: Path, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    main(run_argv(datasets, artifacts))
    run_id = sole_run(artifacts).name
    (artifacts / run_id / "report.json").unlink()
    assert main(["rescore", run_id[:4], "--artifacts", str(artifacts)]) == 0
    assert (artifacts / run_id / "report.json").is_file()


def test_rescore_reports_a_missing_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    assert main(["rescore", "ghost", "--artifacts", str(artifacts)]) == 2
    assert "ghost" in capsys.readouterr().err


def test_rescore_keeps_the_in_process_flag_from_the_manifest(
    datasets: Path, tmp_path: Path
) -> None:
    """A rescore must not silently reclassify a baseline row's provenance."""
    artifacts = tmp_path / "artifacts"
    main(run_argv(datasets, artifacts, "oracle_gold"))
    run_id = sole_run(artifacts).name
    assert main(["rescore", run_id, "--artifacts", str(artifacts)]) == 0
    report = read_json(artifacts / run_id / "report.json")
    assert report["row"]["in_process"] is True


def test_verify_replays_the_checks_over_existing_artifacts(datasets: Path, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    main(run_argv(datasets, artifacts))
    run_id = sole_run(artifacts).name
    (artifacts / run_id / "verify.json").unlink()
    assert main(["verify", run_id, "--artifacts", str(artifacts)]) == 0
    assert (artifacts / run_id / "verify.json").is_file()


def test_verify_reports_a_missing_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    assert main(["verify", "ghost", "--artifacts", str(artifacts)]) == 2
    assert "ghost" in capsys.readouterr().err


def test_verify_exits_nonzero_when_a_run_is_invalidated(
    datasets: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A withheld row must not exit 0, or CI would treat it as a clean run."""
    artifacts = tmp_path / "artifacts"
    main(run_argv(datasets, artifacts))
    run_id = sole_run(artifacts).name
    assert main(["verify", run_id, "--artifacts", str(artifacts), "--metered"]) == 1
    assert "unmetered_model_use" in capsys.readouterr().out


def test_run_expands_a_comma_separated_cross_product(datasets: Path, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    assert main(batch_argv(datasets, artifacts, "bm25,no_memory", "fixture_many")) == 0
    assert (find_run(artifacts, "bm25") / "report.json").is_file()
    assert (find_run(artifacts, "no_memory") / "report.json").is_file()


def test_a_batch_prints_each_generated_id_up_front(
    datasets: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every child's id is known before any work starts, so all of them are stoppable."""
    artifacts = tmp_path / "artifacts"
    main(batch_argv(datasets, artifacts, "bm25,no_memory", "fixture_many"))
    out = capsys.readouterr().out
    for system in ("bm25", "no_memory"):
        run_id = find_run(artifacts, system).name
        assert f"{run_id}  {system}  fixture_many" in out


def test_a_batch_reports_a_failing_run_without_losing_the_others(
    datasets: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifacts = tmp_path / "artifacts"
    code = main(batch_argv(datasets, artifacts, "bm25,ghost", "fixture_many"))
    assert code == 1
    assert (find_run(artifacts, "bm25") / "report.json").is_file()
    assert "ghost" in capsys.readouterr().err


def test_the_same_system_override_marks_rows_contended(datasets: Path, tmp_path: Path) -> None:
    """3.5.1: two runs sharing one system breaks D6(a), so the rows say so."""
    artifacts = tmp_path / "artifacts"
    main(
        batch_argv(
            datasets,
            artifacts,
            "bm25",
            "fixture_many,fixture_single",
            "--allow-concurrent-same-system",
            "--parallel",
            "2",
        )
    )
    report = read_json(find_run(artifacts, "bm25") / "report.json")
    assert report["row"]["contended"] is True
    # These runs had no gateway, so system cost is `unavailable` rather than
    # `unreliable`: there was never a counter to break. The contended flag is what
    # records the D6(a) violation. `test_a_contended_run_records_unreliable_cost`
    # covers the metered case.
    assert report["row"]["cost"]["system_attribution"] == "unavailable"
    assert any("contended" in note for note in report["notes"])


def test_a_solo_run_is_never_contended_even_with_the_override(
    datasets: Path, tmp_path: Path
) -> None:
    """One run cannot contend with itself; a false label would withhold a sound number."""
    artifacts = tmp_path / "artifacts"
    main(run_argv(datasets, artifacts, "bm25", "--allow-concurrent-same-system"))
    assert read_json(sole_run(artifacts) / "report.json")["row"]["contended"] is False


def test_run_accepts_the_contended_mark_from_a_parent_process(
    datasets: Path, tmp_path: Path
) -> None:
    """A child run is told it shares its system; it cannot infer that alone (§3.5.3)."""
    artifacts = tmp_path / "artifacts"
    assert main(run_argv(datasets, artifacts, "bm25", "--contended")) == 0
    assert read_json(sole_run(artifacts) / "report.json")["row"]["contended"] is True


def test_a_batch_runs_each_job_in_its_own_process(datasets: Path, tmp_path: Path) -> None:
    """§3.5.3: one process per run, so a crash cannot take the siblings down."""
    artifacts = tmp_path / "artifacts"
    assert main(batch_argv(datasets, artifacts, "bm25,no_memory", "fixture_many")) == 0
    for system in ("bm25", "no_memory"):
        state = read_json(find_run(artifacts, system) / "state.json")
        assert state["pid"] != os.getpid(), "a batch run must not share the parent's pid"


def test_a_solo_run_stays_in_the_parent_process(datasets: Path, tmp_path: Path) -> None:
    """One run needs no fan-out, and a child would only slow it down and hide its output."""
    artifacts = tmp_path / "artifacts"
    main(run_argv(datasets, artifacts))
    assert read_json(sole_run(artifacts) / "state.json")["pid"] == os.getpid()


class TestResumeByIdAlone:
    """`bench run --resume <id>` needs no system, dataset, or limit (§4.5).

    The manifest already records the configuration, so retyping it can only match or be
    rejected. These drive the CLI the way a user does, including the case where an
    omitted `--limit` used to replan the run at a different size.
    """

    def resume_argv(self, artifacts: Path, run_id: str, datasets: Path) -> list[str]:
        return [
            "run",
            "--resume",
            run_id,
            "--foreground",
            "--artifacts",
            str(artifacts),
            "--datasets",
            str(datasets),
            "--repo-root",
            str(REPO),
        ]

    def test_resuming_needs_neither_system_nor_dataset(
        self, datasets: Path, tmp_path: Path
    ) -> None:
        artifacts = tmp_path / "artifacts"
        assert main(run_argv(datasets, artifacts)) == 0
        run_id = sole_run(artifacts).name

        assert main(self.resume_argv(artifacts, run_id, datasets)) == 0
        assert read_json(sole_run(artifacts) / "manifest.json")["system"]["name"] == "bm25"

    def test_the_recorded_limit_survives_a_resume(self, datasets: Path, tmp_path: Path) -> None:
        """Omitting `--limit` must not replan the run at full size."""
        artifacts = tmp_path / "artifacts"
        assert main(run_argv(datasets, artifacts, "bm25", "--limit", "1")) == 0
        run_dir = sole_run(artifacts)
        planned = (run_dir / "plan.jsonl").read_text().count("\n")

        assert main(self.resume_argv(artifacts, run_dir.name, datasets)) == 0
        assert read_json(run_dir / "manifest.json")["limit"] == 1
        assert (run_dir / "plan.jsonl").read_text().count("\n") == planned

    def test_a_conflicting_system_is_still_refused(self, datasets: Path, tmp_path: Path) -> None:
        """Passing a system that disagrees with the manifest must fail, not be ignored."""
        artifacts = tmp_path / "artifacts"
        assert main(run_argv(datasets, artifacts)) == 0
        run_id = sole_run(artifacts).name

        argv = self.resume_argv(artifacts, run_id, datasets)
        assert main([*argv, "no_memory"]) == 2

    def test_an_unknown_id_is_reported(self, datasets: Path, tmp_path: Path) -> None:
        artifacts = tmp_path / "artifacts"
        assert main(run_argv(datasets, artifacts)) == 0
        assert main(self.resume_argv(artifacts, "nope", datasets)) == 2

    def test_a_system_is_required_without_resume(
        self, datasets: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Omitting both is only meaningful for a resume; a fresh run has nothing to read."""
        artifacts = tmp_path / "artifacts"
        assert (
            main(
                [
                    "run",
                    "--foreground",
                    "--artifacts",
                    str(artifacts),
                    "--datasets",
                    str(datasets),
                    "--repo-root",
                    str(REPO),
                ]
            )
            == 2
        )
        assert "--resume" in capsys.readouterr().err
