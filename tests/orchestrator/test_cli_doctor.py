"""`bench doctor` reports pass/fail per check and exits non-zero on any failure (§1.6).

Doctor's target is an adapter loaded from the registry, so these tests point registry
entries at `reference.fakes:FakeMemory` and drive its flags through `config`. That is the
same path a real system takes — nothing here is a doctor-only shortcut.
"""

from pathlib import Path

import pytest

from orchestrator import cli

FAKE_ENTRY = """
name: fakesys
adapter: reference.fakes:FakeMemory
"""


def entry(root: Path, name: str, body: str) -> Path:
    root.mkdir(exist_ok=True)
    path = root / f"{name}.yaml"
    path.write_text(body)
    return root


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    return entry(tmp_path / "registry", "fakesys", FAKE_ENTRY)


def flawed(root: Path, flaw: str) -> Path:
    """A registry entry for a fake that breaks exactly one part of the contract."""
    return entry(
        root / "registry",
        "fakesys",
        f"name: fakesys\nadapter: reference.fakes:FakeMemory\nconfig: {{{flaw}: true}}\n",
    )


def test_passes_against_a_conformant_adapter(
    registry: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["doctor", "fakesys", "--registry", str(registry)])
    out = capsys.readouterr().out
    assert code == 0
    assert "PASS" in out
    assert "FAIL" not in out


def test_prints_one_row_per_check(registry: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from contract.conformance.checks import CHECKS

    cli.main(["doctor", "fakesys", "--registry", str(registry)])
    out = capsys.readouterr().out
    for check in CHECKS:
        assert check.name in out


def test_a_baseline_needs_no_registry_entry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """In-repo baselines are adapters like any other, addressable by name."""
    code = cli.main(["doctor", "bm25", "--registry", str(tmp_path / "empty")])
    assert code == 0
    assert "PASS" in capsys.readouterr().out


def test_fails_non_zero_when_a_check_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = flawed(tmp_path, "leak_across_users")
    code = cli.main(["doctor", "fakesys", "--registry", str(root)])
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL" in out
    assert "no_cross_user_leakage" in out


def test_reports_the_failure_reason(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = flawed(tmp_path, "overshoot_top_k")
    cli.main(["doctor", "fakesys", "--registry", str(root)])
    assert "top_k" in capsys.readouterr().out


def test_an_unimportable_adapter_fails_with_the_import_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = entry(
        tmp_path / "registry",
        "ghost",
        "name: ghost\nadapter: nonexistent_sdk_xyz:Adapter\n",
    )
    code = cli.main(["doctor", "ghost", "--registry", str(root)])
    assert code == 2
    assert "nonexistent_sdk_xyz" in capsys.readouterr().err


def test_a_missing_registry_field_fails_before_constructing_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = entry(tmp_path / "registry", "broken", "name: broken\n")
    code = cli.main(["doctor", "broken", "--registry", str(root)])
    assert code == 2
    assert "adapter" in capsys.readouterr().err


def test_an_unset_credential_fails_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hosted system's credential is resolved at construction, not mid-ingest."""
    monkeypatch.delenv("FAKESYS_TOKEN", raising=False)
    root = entry(
        tmp_path / "registry",
        "hosted",
        "name: hosted\nadapter: adapters.http:HttpAdapter\n"
        "config:\n  base_url: http://hosted.invalid\n"
        "  auth: {scheme: bearer, key_env: FAKESYS_TOKEN}\n",
    )
    code = cli.main(["doctor", "hosted", "--registry", str(root)])
    assert code == 2
    assert "FAKESYS_TOKEN" in capsys.readouterr().err


def test_never_prints_the_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FAKESYS_TOKEN", "s3cret")
    root = entry(
        tmp_path / "registry",
        "hosted",
        "name: hosted\nadapter: adapters.http:HttpAdapter\n"
        "config:\n  base_url: http://hosted.invalid\n"
        "  auth: {scheme: bearer, key_env: FAKESYS_TOKEN}\n",
    )
    cli.main(["doctor", "hosted", "--registry", str(root)])
    captured = capsys.readouterr()
    assert "s3cret" not in captured.out + captured.err


def test_an_adapter_not_pointed_at_the_gateway_warns_without_failing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """We cannot prove where a library sends traffic, only that it was never told to.

    So this is a warning and an `unenforced` cost column, never a failed check: failing
    would make an honest self-contained system unbenchmarkable.
    """
    root = entry(
        tmp_path / "registry",
        "unmetered",
        "name: unmetered\nadapter: reference.fakes:FakeMemory\n",
    )
    code = cli.main(["doctor", "unmetered", "--registry", str(root)])
    captured = capsys.readouterr()
    assert code == 0
    assert "warning" in captured.err
    assert "unenforced" in captured.err


def test_unknown_command_exits_non_zero() -> None:
    with pytest.raises(SystemExit) as err:
        cli.main(["frobnicate"])
    assert err.value.code != 0
