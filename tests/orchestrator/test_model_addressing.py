"""How a pinned model is addressed on the wire (D11, ARCH §6.1).

Both facts here were established against a running `maximhq/bifrost` rather than from the
docs, because both fail the same way — an HTTP error on the first model call, after the
run has been planned and ingest has been paid for.

- The gateway serves `/v1/chat/completions`, and the chat client appends that path, so the
  base URL must not also carry `/v1`. It answers 405 to `/v1/v1/chat/completions`.
- A model is addressed as `<provider>/<id>`. A bare `deepseek/deepseek-v4-flash-0731` is
  read as provider `deepseek`, which does not exist: 400 `failed to get config for
  provider deepseek`.
"""

from pathlib import Path

import httpx
import pytest

from orchestrator.cli import DEFAULT_MODELS_URL, build_chat
from orchestrator.gateway import PinnedModel, wire_name
from orchestrator.suite import load_suite

REPO = Path(__file__).resolve().parents[2]


def test_the_default_base_url_carries_no_version_prefix() -> None:
    """The chat client appends `/v1/...`; a `/v1` here would make it `/v1/v1/...` (405)."""
    assert not DEFAULT_MODELS_URL.rstrip("/").endswith("/v1")


def test_a_pinned_model_is_addressed_provider_first() -> None:
    assert wire_name(PinnedModel("deepseek/deepseek-v4-flash-0731", "proxy")) == (
        "proxy/deepseek/deepseek-v4-flash-0731"
    )


def test_a_legacy_model_without_a_provider_stays_bare() -> None:
    """v1 predates providers, so its rows are reproduced with the name they used."""
    assert wire_name(PinnedModel("gpt-4o-mini")) == "gpt-4o-mini"


@pytest.mark.parametrize("role", ["chat", "judge", "embedding"])
def test_every_current_suite_model_addresses_its_own_provider(role: str) -> None:
    suite = load_suite("v3", REPO)
    model = getattr(suite.models, role)
    assert wire_name(PinnedModel(model.id, model.provider)).startswith(f"{model.provider}/")


async def test_a_chat_call_reaches_the_unprefixed_completions_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one path the live gateway answers, asserted end to end through build_chat."""
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    monkeypatch.setattr("orchestrator.cli.model_transport", lambda: httpx.MockTransport(handle))
    chat = build_chat()
    await chat.chat([{"role": "user", "content": "hi"}], model="m", temperature=0)
    assert seen == ["/v1/chat/completions"]
