"""`Mem0Adapter` wraps the self-hosted mem0 server's API (§4, §10).

mem0's REST server is started separately and speaks mem0's own API, so this adapter
translates. It is deliberately *not* `HttpAdapter`: that one speaks the harness's
retired `/add`+`/search` wire contract, and mem0's server shares only the path name
`/search` with it — never the payload.

Transport is a local `httpx.MockTransport`; no test here opens a socket.
"""

import json
from typing import Any

import httpx
import pytest

from adapters.mem0 import Mem0Adapter
from contract.adapter import AdapterError, ContractViolation, Message

UID = "eval:r1:fixture:ctx0"
MESSAGES = [Message(role="user", content="a cat named Mochi", timestamp=1704067200000)]


def adapter_for(handler: Any) -> Mem0Adapter:
    """Return an adapter whose transport is a local handler, never a socket."""
    return Mem0Adapter(
        base_url="http://mem0.invalid:8888",
        transport=httpx.MockTransport(handler),
    )


def replying(payload: Any, status: int = 200) -> Any:
    """A handler that records the request it saw and replays `payload`."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content) if request.content else {}
        return httpx.Response(status, json=payload)

    handler.seen = seen  # type: ignore[attr-defined]
    return handler


async def test_add_posts_to_the_memories_endpoint() -> None:
    """mem0 stores at `/memories`, not the harness's retired `/add`."""
    handler = replying({"results": []})
    await adapter_for(handler).add(UID, MESSAGES, chunk_id="c7")
    assert handler.seen["path"] == "/memories"


async def test_add_passes_user_id_as_the_isolation_key() -> None:
    handler = replying({"results": []})
    await adapter_for(handler).add(UID, MESSAGES, chunk_id="c7")
    assert handler.seen["body"]["user_id"] == UID


async def test_add_carries_the_chunk_id_in_metadata() -> None:
    """Without this the row still scores quality, but recall reads `unavailable`."""
    handler = replying({"results": []})
    await adapter_for(handler).add(UID, MESSAGES, chunk_id="c7")
    assert handler.seen["body"]["metadata"] == {"x_chunk_id": "c7"}


async def test_add_sends_messages_as_role_and_content_only() -> None:
    """The server's Message model is `{role, content}`; a timestamp would be dropped."""
    handler = replying({"results": []})
    await adapter_for(handler).add(UID, MESSAGES, chunk_id="c7")
    assert handler.seen["body"]["messages"] == [{"role": "user", "content": "a cat named Mochi"}]


async def test_add_accepts_an_empty_result_list() -> None:
    """mem0 may extract no fact from a chunk. That is its decision, not a failure."""
    await adapter_for(replying({"results": []})).add(UID, MESSAGES, chunk_id="c7")


async def test_search_posts_the_query_verbatim_to_search() -> None:
    """Rewriting the query would measure our rewriting rather than the system."""
    handler = replying({"results": []})
    await adapter_for(handler).search(UID, "who is Mochi?", top_k=5)
    assert handler.seen["path"] == "/search"
    assert handler.seen["body"]["query"] == "who is Mochi?"


async def test_search_scopes_the_user_inside_filters() -> None:
    """A top-level `user_id` is deprecated on this endpoint; `filters` is the live form."""
    handler = replying({"results": []})
    await adapter_for(handler).search(UID, "cat", top_k=5)
    assert handler.seen["body"]["filters"] == {"user_id": UID}


async def test_search_passes_top_k() -> None:
    handler = replying({"results": []})
    await adapter_for(handler).search(UID, "cat", top_k=5)
    assert handler.seen["body"]["top_k"] == 5


async def test_search_maps_memory_to_content_and_echoes_source_ids() -> None:
    handler = replying(
        {
            "results": [
                {
                    "id": "m1",
                    "memory": "a cat named Mochi",
                    "score": 0.9,
                    "metadata": {"x_chunk_id": "c7"},
                }
            ]
        }
    )
    found = await adapter_for(handler).search(UID, "cat", top_k=5)
    assert found[0].content == "a cat named Mochi"
    assert found[0].score == 0.9
    assert list(found[0].source_ids) == ["c7"]


async def test_search_preserves_rank_order() -> None:
    """Rank is the position in the sequence, so the server's order must survive."""
    handler = replying(
        {"results": [{"memory": "first", "score": 0.9}, {"memory": "second", "score": 0.4}]}
    )
    found = await adapter_for(handler).search(UID, "x", top_k=5)
    assert [m.content for m in found] == ["first", "second"]


@pytest.mark.parametrize(
    "payload",
    [
        {"results": [{"id": "m1", "memory": "first"}, {"id": "m2", "memory": "second"}]},
        [{"id": "m1", "memory": "first"}, {"id": "m2", "memory": "second"}],
    ],
    ids=["results_dict", "bare_list"],
)
async def test_both_mem0_reply_shapes_are_accepted(payload: Any) -> None:
    """v1.1 returns {"results": [...]}; older builds returned a bare list."""
    found = await adapter_for(replying(payload)).search(UID, "x", top_k=5)
    assert [m.content for m in found] == ["first", "second"]


async def test_a_memory_without_metadata_reports_no_source_ids() -> None:
    handler = replying({"results": [{"id": "m1", "memory": "text"}]})
    found = await adapter_for(handler).search(UID, "x", top_k=5)
    assert found[0].source_ids == ()


async def test_a_missing_score_is_none_rather_than_zero() -> None:
    """Zero is a score the system could have reported; absent is not."""
    handler = replying({"results": [{"id": "m1", "memory": "text"}]})
    found = await adapter_for(handler).search(UID, "x", top_k=5)
    assert found[0].score is None


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_an_unwell_server_is_retryable(status: int) -> None:
    with pytest.raises(AdapterError):
        await adapter_for(replying({"detail": "nope"}, status=status)).search(UID, "x", top_k=5)


@pytest.mark.parametrize("status", [400, 401, 404, 422])
async def test_a_refused_request_is_a_contract_violation(status: int) -> None:
    """Retrying a rejected request just spends money to be rejected again."""
    with pytest.raises(ContractViolation):
        await adapter_for(replying({"detail": "nope"}, status=status)).add(
            UID, MESSAGES, chunk_id="c7"
        )


async def test_a_transport_failure_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(AdapterError):
        await adapter_for(handler).add(UID, MESSAGES, chunk_id="c7")


async def test_a_non_json_body_is_a_contract_violation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>proxy error</html>")

    with pytest.raises(ContractViolation):
        await adapter_for(handler).search(UID, "x", top_k=5)


async def test_a_memory_without_content_is_a_contract_violation() -> None:
    """Empty content would score as a wrong answer rather than a broken integration."""
    handler = replying({"results": [{"id": "m1", "score": 0.5}]})
    with pytest.raises(ContractViolation):
        await adapter_for(handler).search(UID, "x", top_k=5)


async def test_the_run_key_is_posted_to_configure() -> None:
    """The run-scoped virtual key is what labels mem0's traffic in the gateway's metrics.

    Delivered through `/configure` rather than the container's env: the key is named for a
    run that does not exist when the container starts, and mem0 reads
    `config.api_key or os.getenv(...)`, so config wins without a restart.
    """
    handler = replying({"message": "Configuration set successfully"})
    await adapter_for(handler).use_model_key("vk-run-value")
    assert handler.seen["path"] == "/configure"


async def test_both_halves_of_mem0_are_given_the_key() -> None:
    """Extraction and embedding are separately configured and each reads its own key."""
    handler = replying({"message": "ok"})
    await adapter_for(handler).use_model_key("vk-run-value")
    body = handler.seen["body"]
    assert body["llm"]["config"]["api_key"] == "vk-run-value"
    assert body["embedder"]["config"]["api_key"] == "vk-run-value"


async def test_the_configure_payload_names_no_provider() -> None:
    """A partial config deep-merges; naming a provider would re-validate and can 400.

    It would also risk replacing the pinned model block that `setup_mem0.py` established.
    """
    handler = replying({"message": "ok"})
    await adapter_for(handler).use_model_key("vk-run-value")
    body = handler.seen["body"]
    assert "provider" not in body["llm"]
    assert "provider" not in body["embedder"]
    assert set(body) == {"llm", "embedder"}
    assert set(body["llm"]["config"]) == {"api_key"}


async def test_a_refused_configure_is_a_contract_violation() -> None:
    """A system that cannot be metered must not quietly produce an unattributable run."""
    handler = replying({"detail": "nope"}, status=400)
    with pytest.raises(ContractViolation):
        await adapter_for(handler).use_model_key("vk-run-value")


async def test_a_configure_outage_is_retryable() -> None:
    handler = replying({"detail": "unavailable"}, status=503)
    with pytest.raises(AdapterError):
        await adapter_for(handler).use_model_key("vk-run-value")
