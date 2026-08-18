"""Every model call goes through the gateway, and there is no way to say otherwise (§3, D5).

The gateway is the only route to a model, so it is not a choice: no endpoint flag, no
credential flag, and no stub. A run works once `bench gateway` is up.

These tests pin the wiring; they never make a real model call.
"""

from pathlib import Path

import httpx
import pytest

from orchestrator.cli import DEFAULT_MODELS_URL, MODELS_KEY_ENV, build_chat, main


def test_the_default_endpoint_is_the_local_gateway() -> None:
    """The gateway's published port is where model calls go, with no flag to say so.

    The path is the chat client's business: it appends `/v1/chat/completions`, so this is
    the service root. See `test_model_addressing.py`.
    """
    assert DEFAULT_MODELS_URL == "http://127.0.0.1:8080"


def test_a_chat_client_needs_no_configuration_at_all() -> None:
    """The whole point: a run works after `bench gateway`, with no model flags."""
    chat = build_chat()
    assert hasattr(chat, "chat")


def test_there_is_no_stub_models_flag() -> None:
    """Removed, not deprecated: a stale script must fail rather than fake a run.

    A stubbed run reported 1.000 accuracy for every system, including the `no_memory`
    floor, and zero cost — a report shaped exactly like a measured one but measuring
    nothing. `bench doctor` covers adapter conformance without that hazard.
    """
    with pytest.raises(SystemExit):
        main(["run", "bm25", "--dataset", "fixture_many", "--stub-models"])


def test_the_credential_is_read_from_one_fixed_variable() -> None:
    """One endpoint means one key, so its name is a constant rather than a flag."""
    assert MODELS_KEY_ENV == "ENGRAM_GATEWAY_KEY"


def test_an_unset_credential_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gateway holds the provider keys; a local one may need no client credential.

    Refusing here would block the common case — `bench gateway` up, governance off, no
    virtual key — and the gateway's own 401 is a better error than a guess made now.
    """
    monkeypatch.delenv(MODELS_KEY_ENV, raising=False)
    chat = build_chat()
    assert hasattr(chat, "chat")


def test_the_credential_never_appears_in_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLAUDE.md: never log API keys. A traceback carrying one is a leak."""
    monkeypatch.setenv(MODELS_KEY_ENV, "sk-secret-value")
    build_chat()
    captured = capsys.readouterr()
    assert "sk-secret-value" not in captured.out + captured.err


@pytest.mark.parametrize("flag", ["--models-url", "--models-key-env"])
def test_the_removed_model_flags_are_rejected(
    flag: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Removed rather than deprecated: a stale script must fail loudly, not route away."""
    with pytest.raises(SystemExit):
        main(["run", "bm25", "--dataset", "fixture_many", flag, "x"])
    assert "unrecognized arguments" in capsys.readouterr().err


def test_an_unreachable_gateway_is_reported_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The likeliest failure now that the gateway is the default: it is not running."""
    import shutil

    repo = Path(__file__).resolve().parents[2]
    datasets = tmp_path / "datasets"
    shutil.copytree(repo / "tests" / "datasets" / "fixtures", datasets)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("orchestrator.cli.model_transport", lambda: httpx.MockTransport(refuse))

    code = main(
        [
            "run",
            "bm25",
            # Attached: the refusing transport is patched into *this* process, and the
            # assertion is that this command reports the failure rather than a traceback.
            "--foreground",
            "--dataset",
            "fixture_many",
            "--run-id",
            "r1",
            "--artifacts",
            str(tmp_path / "artifacts"),
            "--datasets",
            str(datasets),
            "--repo-root",
            str(repo),
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    # Names the fix, since there is no flag to point somewhere else.
    assert "bench gateway" in err


def test_a_real_run_sends_its_prompts_to_the_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Answers come from a model, so accuracy can differ from the stub's 1.000."""
    import shutil

    from orchestrator.artifacts import read_json

    repo = Path(__file__).resolve().parents[2]
    datasets = tmp_path / "datasets"
    shutil.copytree(repo / "tests" / "datasets" / "fixtures", datasets)

    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url.path))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"label": "WRONG"}'}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            },
        )

    monkeypatch.setattr("orchestrator.cli.model_transport", lambda: httpx.MockTransport(handle))

    artifacts = tmp_path / "artifacts"
    code = main(
        [
            "run",
            "bm25",
            # Attached: the recording transport lives in this process, so a detached child
            # would call a real endpoint and `seen` would stay empty.
            "--foreground",
            "--dataset",
            "fixture_many",
            "--run-id",
            "r1",
            "--artifacts",
            str(artifacts),
            "--datasets",
            str(datasets),
            "--repo-root",
            str(repo),
        ]
    )
    assert code == 0
    assert seen, "no model call was made"

    row = read_json(artifacts / "r1" / "report.json")["row"]
    # A judge that answers WRONG must produce 0.0, not the stub's 1.000.
    assert row["quality"]["accuracy"] == 0.0
    assert row["cost"]["harness_prompt_tokens"] > 0
