"""A broken adapter must only break the commands that need it (ARCH §14, Phase 6).

The cost of importing memory systems into our interpreter (D8) is that one system's
missing or conflicting SDK is now an `ImportError` in our process. Lazy importing is what
bounds that: `load_system` reads YAML and nothing else, so the import happens when a
command actually needs to *call* the system.

Everything offline stays working — `ps` reads `state.json`, `rescore` and `verify` read
artifacts, and none of them touch a memory system. If a dead SDK could take those down,
a finished run would become unreadable because of a package it no longer needs.
"""

import shutil
from pathlib import Path

import pytest

from orchestrator.artifacts import read_json
from orchestrator.cli import main

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "datasets" / "fixtures"

# Names a module that does not exist, so constructing it is an unavoidable ImportError.
BROKEN = """
name: broken
adapter: no_such_sdk_at_all:Adapter
"""


@pytest.fixture
def datasets(tmp_path: Path) -> Path:
    dest = tmp_path / "datasets"
    shutil.copytree(FIXTURES, dest)
    return dest


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    root = tmp_path / "registry"
    root.mkdir()
    (root / "broken.yaml").write_text(BROKEN)
    return root


@pytest.fixture
def finished_run(datasets: Path, tmp_path: Path, registry: Path) -> tuple[Path, str]:
    """One completed run's artifacts, produced by a system that still works."""
    artifacts = tmp_path / "artifacts"
    code = main(
        [
            "run",
            "bm25",
            # Attached: this fixture's whole purpose is to hand back a *finished* run.
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
            "--registry",
            str(registry),
        ]
    )
    assert code == 0
    run_id = next(p for p in sorted(artifacts.iterdir()) if (p / "manifest.json").is_file()).name
    return artifacts, run_id


def test_ps_works_while_a_registered_adapter_cannot_be_imported(
    finished_run: tuple[Path, str], registry: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `ps` takes no `--registry` at all, which is the strongest form of the property:
    # it reads `state.json` and cannot reach a registry entry to break on.
    artifacts, run_id = finished_run
    assert main(["ps", "--all", "--artifacts", str(artifacts)]) == 0
    assert run_id in capsys.readouterr().out


def test_rescore_works_while_a_registered_adapter_cannot_be_imported(
    finished_run: tuple[Path, str], registry: Path
) -> None:
    artifacts, run_id = finished_run
    before = (artifacts / run_id / "report.json").read_bytes()
    (artifacts / run_id / "report.json").unlink()
    assert main(["rescore", run_id, "--artifacts", str(artifacts)]) == 0
    assert (artifacts / run_id / "report.json").read_bytes() == before


def test_verify_works_while_a_registered_adapter_cannot_be_imported(
    finished_run: tuple[Path, str],
) -> None:
    artifacts, run_id = finished_run
    (artifacts / run_id / "verify.json").unlink()
    assert main(["verify", run_id, "--artifacts", str(artifacts)]) == 0
    assert (artifacts / run_id / "verify.json").is_file()


def test_loading_the_entry_does_not_import_the_adapter(registry: Path) -> None:
    """The YAML read and the import are separate steps, which is what makes this hold."""
    from orchestrator.registry import load_system

    spec = load_system("broken", registry)
    assert spec.adapter == "no_such_sdk_at_all:Adapter"


def test_only_running_the_broken_system_fails_and_it_names_the_module(
    datasets: Path, tmp_path: Path, registry: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The failure lands on the command that needed the system, naming what was missing."""
    code = main(
        [
            "run",
            "broken",
            # Attached, so the import failure is this process's to report. A detached run
            # records the same failure in its own `run.log` instead.
            "--foreground",
            "--dataset",
            "fixture_many",
            "--suite",
            "v1",
            "--artifacts",
            str(tmp_path / "artifacts"),
            "--datasets",
            str(datasets),
            "--repo-root",
            str(REPO),
            "--registry",
            str(registry),
        ]
    )
    assert code == 2
    assert "no_such_sdk_at_all" in capsys.readouterr().err


def test_a_run_by_a_working_system_is_unaffected(finished_run: tuple[Path, str]) -> None:
    artifacts, run_id = finished_run
    assert read_json(artifacts / run_id / "report.json")["row"]["system"] == "bm25"
