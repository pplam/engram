"""`bench systems` answers "what can I pass to `bench run`?" (§3.5, §10, §11).

Three kinds of name are runnable and they come from three different places: registry YAML
under `registry/`, the two in-repo baselines in `BASELINE_ADAPTERS`, and the two
privileged in-process baselines in `run.IN_PROCESS`. Nothing listed all three, so the only
way to learn a name was to read the source or guess and read the error.

The listing imports nothing. `load_system` reads YAML and stops, which is what lets this
command describe a system whose SDK is absent — the same property `bench ps` relies on
(ARCH §14). A listing that constructed adapters would fail on exactly the entry a user is
trying to diagnose.
"""

import json
from pathlib import Path

import pytest

from orchestrator.cli import main

REPO = Path(__file__).resolve().parents[2]


def registry(tmp_path: Path, **entries: str) -> Path:
    """Write a registry directory holding one YAML file per entry."""
    root = tmp_path / "registry"
    root.mkdir()
    for name, adapter in entries.items():
        (root / f"{name}.yaml").write_text(f"name: {name}\nadapter: {adapter}\n")
    return root


def test_systems_lists_a_registered_entry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = registry(tmp_path, mysystem="adapters.mem0:Mem0Adapter")
    assert main(["systems", "--registry", str(root)]) == 0
    assert "mysystem" in capsys.readouterr().out


def test_systems_lists_the_baselines(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A baseline is runnable without a registry entry, so a registry listing would hide it."""
    out = capsys.readouterr
    main(["systems", "--registry", str(registry(tmp_path))])
    text = out().out
    for name in ("no_memory", "bm25"):
        assert name in text


def test_systems_lists_the_privileged_baselines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`oracle_gold` and `long_context` are valid `--system` values but have no adapter.

    They bypass the interface and emit `retrieve.jsonl` directly (D2), so they appear in
    neither the registry nor `BASELINE_ADAPTERS` — and were the hardest names to discover.
    """
    main(["systems", "--registry", str(registry(tmp_path))])
    text = capsys.readouterr().out
    for name in ("oracle_gold", "long_context"):
        assert name in text


def test_systems_says_where_each_name_comes_from(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The kind is what tells you whether a row is comparable with a real system's."""
    root = registry(tmp_path, mysystem="adapters.mem0:Mem0Adapter")
    main(["systems", "--registry", str(root)])
    text = capsys.readouterr().out
    assert "registry" in text
    assert "baseline" in text
    assert "privileged" in text


def test_systems_shows_the_adapter_a_registered_entry_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`module:Class` is the one field that says what a registry entry actually runs."""
    root = registry(tmp_path, mysystem="adapters.mem0:Mem0Adapter")
    main(["systems", "--registry", str(root)])
    assert "adapters.mem0:Mem0Adapter" in capsys.readouterr().out


def test_systems_imports_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An entry naming an absent module must still be listed, not crash the listing.

    This is the case the command exists for: a user whose SDK is missing needs to see the
    entry to diagnose it. `bench doctor` is what imports and reports the failure.
    """
    root = registry(tmp_path, broken="no_such_module:NoSuchAdapter")
    assert main(["systems", "--registry", str(root)]) == 0
    assert "broken" in capsys.readouterr().out


def test_systems_reports_an_unreadable_entry_without_hiding_the_others(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One malformed file must not blank the listing, the way `bench ps` degrades a row."""
    root = registry(tmp_path, good="adapters.mem0:Mem0Adapter")
    (root / "bad.yaml").write_text("name: mismatched\nadapter: a:B\n")
    assert main(["systems", "--registry", str(root)]) == 0
    text = capsys.readouterr().out
    assert "good" in text
    assert "bad" in text


def test_systems_emits_json_on_request(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = registry(tmp_path, mysystem="adapters.mem0:Mem0Adapter")
    assert main(["systems", "--registry", str(root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    entry = next(row for row in payload if row["name"] == "mysystem")
    assert entry["kind"] == "registry"
    assert entry["adapter"] == "adapters.mem0:Mem0Adapter"


def test_systems_finds_the_repos_own_registry_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default has to be the real registry, or the command needs a flag to be useful."""
    assert main(["systems", "--registry", str(REPO / "registry")]) == 0
    assert "mem0" in capsys.readouterr().out


def test_systems_sorts_within_kind_so_the_listing_is_stable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Directory order is filesystem-dependent; a listing that reorders is hard to diff."""
    root = registry(
        tmp_path,
        zebra="adapters.mem0:Mem0Adapter",
        alpha="adapters.mem0:Mem0Adapter",
    )
    main(["systems", "--registry", str(root), "--json"])
    names = [
        row["name"] for row in json.loads(capsys.readouterr().out) if row["kind"] == "registry"
    ]
    assert names == sorted(names)
