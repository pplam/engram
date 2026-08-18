"""`HttpAdapter` is the bridge for a system that is a container or a hosted service.

It is the one adapter the harness ships that still speaks HTTP, so it has to reproduce
the old client's wire behaviour exactly — every currently-registered system falls back
to it. Transport is a local `httpx.MockTransport`; no test here opens a socket.
"""

from typing import Any

import httpx
import pytest

from adapters.http import HttpAdapter
from contract.adapter import AdapterError, ContractViolation, Message

UID = "eval:r1:fixture:ctx0"
MESSAGES = [Message(role="user", content="the cat sat on the mat", timestamp=1704067200000)]


def adapter_for(handler: Any, **kwargs: Any) -> HttpAdapter:
    """Return an adapter whose transport is a local handler, never a socket."""
    return HttpAdapter(
        base_url="http://system.invalid",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


async def test_add_posts_the_messages_to_the_add_endpoint() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = httpx.Request("POST", request.url, content=request.content).read()
        return httpx.Response(200, json={"success": True})

    await adapter_for(handler).add(UID, MESSAGES, chunk_id="c0")
    assert seen["path"] == "/add"


async def test_add_sends_user_id_chunk_id_and_message_bodies() -> None:
    import json

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"success": True})

    await adapter_for(handler).add(UID, MESSAGES, chunk_id="c0")
    assert captured["user_id"] == UID
    assert captured["x_chunk_id"] == "c0"
    assert captured["messages"] == [
        {"role": "user", "content": "the cat sat on the mat", "timestamp": 1704067200000}
    ]


async def test_search_returns_memories_mapped_from_the_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "m1",
                        "content": "a cat named Mochi",
                        "score": 0.9,
                        "x_source_ids": ["c7"],
                    }
                ]
            },
        )

    found = await adapter_for(handler).search(UID, "cat", top_k=5)
    assert len(found) == 1
    assert found[0].content == "a cat named Mochi"
    assert found[0].score == 0.9
    assert list(found[0].source_ids) == ["c7"]


async def test_search_preserves_the_order_the_system_returned() -> None:
    """Rank is position, so reordering here would silently change every retrieval metric."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "m2", "content": "second best", "score": 0.1},
                    {"id": "m1", "content": "best", "score": 0.9},
                ]
            },
        )

    found = await adapter_for(handler).search(UID, "x", top_k=5)
    assert [m.content for m in found] == ["second best", "best"]


async def test_a_memory_without_source_ids_reports_an_empty_tuple() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "m1", "content": "text"}]})

    found = await adapter_for(handler).search(UID, "x", top_k=5)
    assert found[0].source_ids == ()
    assert found[0].score is None


async def test_a_transport_error_becomes_a_retryable_adapter_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(AdapterError, match="transport"):
        await adapter_for(handler).add(UID, MESSAGES, chunk_id="c0")


@pytest.mark.parametrize("status", [500, 502, 503, 429])
async def test_a_retryable_status_becomes_an_adapter_error(status: int) -> None:
    """The adapter classifies; `client.py` owns whether to retry (§2.6)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "upstream"})

    with pytest.raises(AdapterError, match=str(status)):
        await adapter_for(handler).add(UID, MESSAGES, chunk_id="c0")


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_a_client_error_is_a_contract_violation(status: int) -> None:
    """A 4xx is our request being wrong or the system rejecting us: never retry it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "nope"})

    with pytest.raises(ContractViolation, match=str(status)):
        await adapter_for(handler).add(UID, MESSAGES, chunk_id="c0")


async def test_a_non_json_body_is_a_contract_violation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all")

    with pytest.raises(ContractViolation, match="JSON"):
        await adapter_for(handler).search(UID, "x", top_k=5)


async def test_add_reporting_success_false_is_a_contract_violation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"success": False})

    with pytest.raises(ContractViolation, match="success"):
        await adapter_for(handler).add(UID, MESSAGES, chunk_id="c0")


async def test_a_search_response_without_a_data_list_is_a_contract_violation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    with pytest.raises(ContractViolation, match="data"):
        await adapter_for(handler).search(UID, "x", top_k=5)


async def test_a_memory_missing_content_is_a_contract_violation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "m1"}]})

    with pytest.raises(ContractViolation, match="content"):
        await adapter_for(handler).search(UID, "x", top_k=5)


async def test_the_auth_credential_is_read_from_its_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYSTEM_TOKEN", "s3cret")
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"success": True})

    adapter = adapter_for(handler, auth={"scheme": "bearer", "key_env": "SYSTEM_TOKEN"})
    await adapter.add(UID, MESSAGES, chunk_id="c0")
    assert seen["authorization"] == "Bearer s3cret"


async def test_an_unset_credential_fails_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYSTEM_TOKEN", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made without the credential")

    with pytest.raises(ContractViolation, match="SYSTEM_TOKEN"):
        adapter_for(handler, auth={"scheme": "bearer", "key_env": "SYSTEM_TOKEN"})


async def test_close_releases_the_client() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"success": True})

    adapter = adapter_for(handler)
    await adapter.add(UID, MESSAGES, chunk_id="c0")
    await adapter.close()
    with pytest.raises(RuntimeError):
        await adapter.add(UID, MESSAGES, chunk_id="c0")
