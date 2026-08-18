"""`bench gateway start` runs the model plane and waits for it; `stop` ends it (§3.3).

The two seams the CLI exposes for this — `compose_launch` and `gateway_transport` — are
patched here, so no test runs `docker` or opens a socket.
"""

import json
from pathlib import Path

import httpx
import pytest

from orchestrator.cli import main
from orchestrator.compose import DEFAULT_URL


@pytest.fixture
def plane(tmp_path: Path) -> tuple[Path, Path]:
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("services:\n  gateway: {}\n")
    config = tmp_path / "bifrost.json"
    config.write_text(json.dumps({"providers": {"openai": {"keys": [{"value": "env.KEY_A"}]}}}))
    return compose, config


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record the argv `bench gateway start`/`stop` would run instead of running it."""
    calls: list[list[str]] = []

    def fake(argv: list[str]) -> int:
        calls.append(argv)
        return 0

    monkeypatch.setattr("orchestrator.cli.compose_launch", fake)
    return calls


def _ready(monkeypatch: pytest.MonkeyPatch, status: int = 200) -> None:
    handler = httpx.MockTransport(lambda request: httpx.Response(status, text=""))
    monkeypatch.setattr("orchestrator.cli.gateway_transport", lambda: handler)


def _argv(plane: tuple[Path, Path], *extra: str) -> list[str]:
    compose, config = plane
    return ["gateway", "start", "--compose", str(compose), "--config", str(config), *extra]


def test_gateway_starts_only_the_gateway_service(
    plane: tuple[Path, Path],
    launched: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KEY_A", "sk-live")
    _ready(monkeypatch)
    assert main(_argv(plane)) == 0
    assert launched == [
        ["docker", "compose", "-f", str(plane[0]), "up", "-d", "gateway"],
    ]


def test_gateway_prints_the_url_to_pass_to_run(
    plane: tuple[Path, Path],
    launched: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("KEY_A", "sk-live")
    _ready(monkeypatch)
    main(_argv(plane))
    # The URL, but no flag to pass it with: `bench run` defaults to the gateway.
    out = capsys.readouterr().out
    assert DEFAULT_URL in out
    assert "--models-url" not in out


def test_gateway_refuses_when_no_credential_is_set(
    plane: tuple[Path, Path],
    launched: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With nothing exported, every provider 401s and no run can work."""
    monkeypatch.delenv("KEY_A", raising=False)
    assert main(_argv(plane)) == 2
    err = capsys.readouterr().err
    assert "KEY_A" in err
    assert launched == []


def test_a_partly_configured_gateway_warns_and_starts(
    tmp_path: Path,
    launched: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One provider per dataset: a missing judge key must not block an OpenAI-only run."""
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("services:\n  gateway: {}\n")
    config = tmp_path / "bifrost.json"
    config.write_text(
        json.dumps(
            {
                "providers": {
                    "openai": {"keys": [{"value": "env.KEY_A"}]},
                    "vllm": {"keys": [{"value": "env.KEY_B"}]},
                }
            }
        )
    )
    monkeypatch.setenv("KEY_A", "sk-live")
    monkeypatch.delenv("KEY_B", raising=False)
    _ready(monkeypatch)
    assert main(["gateway", "start", "--compose", str(compose), "--config", str(config)]) == 0
    err = capsys.readouterr().err
    assert "KEY_B" in err
    assert "KEY_A" not in err
    assert len(launched) == 1


def test_gateway_never_echoes_a_credential_value(
    plane: tuple[Path, Path],
    launched: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("KEY_A", "sk-secret-value")
    _ready(monkeypatch)
    main(_argv(plane))
    captured = capsys.readouterr()
    assert "sk-secret-value" not in captured.out + captured.err


def test_skip_env_check_starts_without_credentials(
    plane: tuple[Path, Path],
    launched: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gateway pointed only at a keyless local endpoint needs no provider key."""
    monkeypatch.delenv("KEY_A", raising=False)
    _ready(monkeypatch)
    assert main(_argv(plane, "--skip-env-check")) == 0
    assert len(launched) == 1


def test_gateway_reports_a_failed_compose_without_waiting(
    plane: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("KEY_A", "sk-live")
    monkeypatch.setattr("orchestrator.cli.compose_launch", lambda argv: 1)
    # No transport is installed: reaching the readiness poll at all would be the bug.
    assert main(_argv(plane)) == 1
    assert "docker compose" in capsys.readouterr().err


def test_gateway_reports_a_missing_docker(
    plane: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def missing(argv: list[str]) -> int:
        raise FileNotFoundError(argv[0])

    monkeypatch.setenv("KEY_A", "sk-live")
    monkeypatch.setattr("orchestrator.cli.compose_launch", missing)
    assert main(_argv(plane)) == 2
    assert "docker" in capsys.readouterr().err


def test_gateway_fails_when_it_never_becomes_ready(
    plane: tuple[Path, Path],
    launched: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("KEY_A", "sk-live")
    _ready(monkeypatch, status=503)
    assert main(_argv(plane, "--timeout", "0")) == 1
    assert "not ready" in capsys.readouterr().err


def test_no_wait_skips_the_readiness_poll(
    plane: tuple[Path, Path],
    launched: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KEY_A", "sk-live")
    # Unreachable base URL and no transport patched: a poll would fail the run.
    assert main(_argv(plane, "--no-wait")) == 0
    assert len(launched) == 1


def test_gateway_reports_an_unreadable_config(
    tmp_path: Path,
    launched: list[list[str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("services: {}\n")
    config = tmp_path / "bifrost.json"
    config.write_text("{not json")
    assert main(["gateway", "start", "--compose", str(compose), "--config", str(config)]) == 2
    assert "bifrost.json" in capsys.readouterr().err
    assert launched == []


def test_gateway_reports_a_missing_compose_file(
    tmp_path: Path,
    launched: list[list[str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["gateway", "start", "--compose", str(tmp_path / "nope.yml")]) == 2
    assert "nope.yml" in capsys.readouterr().err
    assert launched == []


def test_gateway_requires_a_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    """Bare `bench gateway` no longer starts anything: start and stop are both explicit."""
    with pytest.raises(SystemExit) as exit_info:
        main(["gateway"])
    assert exit_info.value.code == 2
    assert "start" in capsys.readouterr().err


def test_stop_stops_only_the_gateway_service(
    plane: tuple[Path, Path],
    launched: list[list[str]],
) -> None:
    compose, _ = plane
    assert main(["gateway", "stop", "--compose", str(compose)]) == 0
    assert launched == [["docker", "compose", "-f", str(compose), "stop", "gateway"]]


def test_stop_needs_no_credentials(
    plane: tuple[Path, Path],
    launched: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stopping reads no config: the credential check exists to catch a bad *start*."""
    monkeypatch.delenv("KEY_A", raising=False)
    assert main(["gateway", "stop", "--compose", str(plane[0])]) == 0
    assert len(launched) == 1


def test_stop_reports_a_failed_compose(
    plane: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("orchestrator.cli.compose_launch", lambda argv: 1)
    assert main(["gateway", "stop", "--compose", str(plane[0])]) == 1
    assert "docker compose" in capsys.readouterr().err


def test_stop_reports_a_missing_docker(
    plane: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def missing(argv: list[str]) -> int:
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr("orchestrator.cli.compose_launch", missing)
    assert main(["gateway", "stop", "--compose", str(plane[0])]) == 2
    assert "docker" in capsys.readouterr().err


def test_stop_reports_a_missing_compose_file(
    tmp_path: Path,
    launched: list[list[str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["gateway", "stop", "--compose", str(tmp_path / "nope.yml")]) == 2
    assert "nope.yml" in capsys.readouterr().err
    assert launched == []


def test_status_reports_a_reachable_gateway(
    launched: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ready(monkeypatch)
    assert main(["gateway", "status"]) == 0
    out = capsys.readouterr().out
    assert "up" in out
    assert DEFAULT_URL in out
    assert launched == [], "status must not start or stop anything"


def test_status_exits_nonzero_when_the_gateway_is_down(
    launched: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Scriptable: `bench gateway status && bench run ...` must not run against nothing."""

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("orchestrator.cli.gateway_transport", lambda: httpx.MockTransport(refused))
    assert main(["gateway", "status"]) == 1
    out = capsys.readouterr().out
    assert "down" in out
    assert launched == []


def test_status_reports_down_when_metrics_errors(
    launched: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A container that is up but 503s `/metrics` cannot meter a run, so it is not up."""
    _ready(monkeypatch, status=503)
    assert main(["gateway", "status"]) == 1
    assert "down" in capsys.readouterr().out


def test_status_names_the_command_that_starts_it(
    launched: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ready(monkeypatch, status=503)
    main(["gateway", "status"])
    assert "bench gateway start" in capsys.readouterr().out


def test_status_honours_an_explicit_url(
    launched: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ready(monkeypatch)
    assert main(["gateway", "status", "--url", "http://gw.example:9000"]) == 0
    assert "http://gw.example:9000" in capsys.readouterr().out
