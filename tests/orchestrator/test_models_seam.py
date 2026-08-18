"""The model seam: one narrow protocol, no test ever making a real call (§2.8)."""

from collections.abc import Callable

import httpx
import pytest

from orchestrator.models import (
    ChatClient,
    Completion,
    ModelError,
    OpenAiCompatibleChat,
    StubChat,
)

Handler = Callable[[httpx.Request], httpx.Response]


def chat_for(
    handler: Handler,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
) -> OpenAiCompatibleChat:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://models.invalid"
    )
    return OpenAiCompatibleChat(client, api_key=api_key, headers=headers)


def completion(text: str, prompt: int = 11, out: int = 3) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": out},
        },
    )


async def test_stub_satisfies_the_protocol() -> None:
    stub: ChatClient = StubChat(replies=["hello"])
    result = await stub.chat([{"role": "user", "content": "hi"}], model="m", temperature=0)
    assert result.text == "hello"


async def test_stub_records_what_it_was_asked() -> None:
    stub = StubChat(replies=["a"])
    await stub.chat([{"role": "user", "content": "hi"}], model="m", temperature=0, seed=7)
    assert stub.calls[0].model == "m"
    assert stub.calls[0].temperature == 0
    assert stub.calls[0].seed == 7
    assert stub.calls[0].messages[0]["content"] == "hi"


async def test_stub_cycles_through_its_replies() -> None:
    stub = StubChat(replies=["a", "b"])
    first = await stub.chat([], model="m", temperature=0)
    second = await stub.chat([], model="m", temperature=0)
    third = await stub.chat([], model="m", temperature=0)
    assert (first.text, second.text, third.text) == ("a", "b", "a")


async def test_stub_reports_token_counts() -> None:
    stub = StubChat(replies=["a"], prompt_tokens=5, completion_tokens=2)
    result = await stub.chat([], model="m", temperature=0)
    assert (result.prompt_tokens, result.completion_tokens) == (5, 2)


async def test_provider_returns_the_message_content_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return completion("Mochi")

    async with chat_for(handler) as chat:
        result = await chat.chat([{"role": "user", "content": "name?"}], model="m", temperature=0)
    assert isinstance(result, Completion)
    assert result.text == "Mochi"
    assert (result.prompt_tokens, result.completion_tokens) == (11, 3)


async def test_provider_sends_the_pinned_model_temperature_and_seed() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(request.content))
        return completion("ok")

    async with chat_for(handler) as chat:
        await chat.chat([{"role": "user", "content": "q"}], model="pinned", temperature=0, seed=7)
    assert seen[0]["model"] == "pinned"
    assert seen[0]["temperature"] == 0
    assert seen[0]["seed"] == 7


async def test_provider_omits_seed_when_unset() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(request.content))
        return completion("ok")

    async with chat_for(handler) as chat:
        await chat.chat([], model="m", temperature=0)
    assert "seed" not in seen[0]


async def test_provider_sends_the_api_key_as_a_bearer_token() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization", ""))
        return completion("ok")

    async with chat_for(handler, api_key="k3y") as chat:
        await chat.chat([], model="m", temperature=0)
    assert seen[0] == "Bearer k3y"


async def test_provider_sends_extra_headers_such_as_gateway_dimensions() -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("x-bf-dim-phase"))
        return completion("ok")

    async with chat_for(handler, headers={"x-bf-dim-phase": "answer"}) as chat:
        await chat.chat([], model="m", temperature=0)
    assert seen[0] == "answer"


async def test_a_virtual_key_can_be_presented_after_construction() -> None:
    """The key is provisioned at run start, after the client already exists.

    Without this the harness reaches the gateway unkeyed, its calls carry no
    `virtual_key_id`, and `verify` withholds the row for metering nothing.
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("x-bf-vk", ""))
        return completion("ok")

    async with chat_for(handler) as chat:
        await chat.use_key("vk-value", run_id="r1", phase="answer")
        await chat.chat([], model="m", temperature=0)
    assert seen[0] == "vk-value"


async def test_presenting_a_key_also_labels_the_traffic_for_prometheus() -> None:
    """`virtual_key_id` alone is decodable; the dimensions make queries legible (§3.1)."""
    seen: list[tuple[str | None, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.headers.get("x-bf-dim-run"), request.headers.get("x-bf-dim-phase")))
        return completion("ok")

    async with chat_for(handler) as chat:
        await chat.use_key("vk-value", run_id="r1", phase="judge")
        await chat.chat([], model="m", temperature=0)
    assert seen[0] == ("r1", "judge")


async def test_a_client_never_given_a_key_sends_no_virtual_key_header() -> None:
    """An unmetered run is a supported configuration, not a degraded keyed one."""
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("x-bf-vk"))
        return completion("ok")

    async with chat_for(handler) as chat:
        await chat.chat([], model="m", temperature=0)
    assert seen[0] is None


async def test_a_non_2xx_response_raises_model_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    async with chat_for(handler) as chat:
        with pytest.raises(ModelError, match="500"):
            await chat.chat([], model="m", temperature=0)


async def test_a_blocked_model_reports_the_pinning_failure() -> None:
    """Bifrost answers 403 model_blocked when a key requests an unpinned model."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"type": "model_blocked"}})

    async with chat_for(handler) as chat:
        with pytest.raises(ModelError, match="model_blocked"):
            await chat.chat([], model="forbidden", temperature=0)


async def test_a_response_without_choices_raises_model_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"usage": {}})

    async with chat_for(handler) as chat:
        with pytest.raises(ModelError, match="choices"):
            await chat.chat([], model="m", temperature=0)


async def test_missing_usage_is_reported_as_zero_not_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async with chat_for(handler) as chat:
        result = await chat.chat([], model="m", temperature=0)
    assert (result.prompt_tokens, result.completion_tokens) == (0, 0)
