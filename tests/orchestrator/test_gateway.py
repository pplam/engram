"""Key provisioning in two scopes, and the phase boundary that makes cost real (§3.2)."""

import json
from typing import Any

import httpx
import pytest

from orchestrator.gateway import (
    PHASES,
    Gateway,
    GatewayError,
    PinnedModel,
    harness_key_name,
    system_key_name,
)
from orchestrator.metrics import Usage

SCRAPE = 'bifrost_input_tokens_total{virtual_key_id="vk_1"} 50\n'


def gateway(handler: Any, base_url: str = "http://gateway.invalid") -> Gateway:
    transport = httpx.MockTransport(handler)
    return Gateway(httpx.AsyncClient(transport=transport, base_url=base_url))


def created_response(vk_id: str, value: str = "sk-bf-x") -> dict[str, Any]:
    """The envelope Bifrost actually returns from a key POST, scraped from a live 1.x image.

    Two things here are not what the API docs describe, and both were wrong in this file
    before: the key object is nested under `virtual_key`, and its identifier is `id`. A mock
    that returns a flat `{"vk_id": ...}` passes while production cannot provision at all,
    which is exactly what happened. Fields the parser ignores are kept so the fixture stays
    recognisable against a real response.
    """
    return {
        "message": "Virtual key created successfully",
        "virtual_key": {
            "id": vk_id,
            "name": "eval:r1:bm25:answer",
            "value": value,
            "is_active": True,
            "provider_configs": [],
        },
    }


def key_handler(created: list[dict[str, Any]]) -> Any:
    """Serve key creation and a scrape, recording each provisioning payload in `created`."""

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/governance/virtual-keys" and request.method == "POST":
            created.append(json.loads(request.content))
            return httpx.Response(200, json=created_response(f"vk_{len(created)}"))
        if path == "/metrics":
            return httpx.Response(200, text=SCRAPE)
        return httpx.Response(404, json={"error": "not_found"})

    return handle


def test_key_names_are_scoped_per_phase_for_the_harness() -> None:
    """D3: harness calls are ours, so attribution is exact per phase."""
    assert harness_key_name("r1", "bm25", "answer") == "eval:r1:bm25:answer"


def test_a_system_gets_one_key_for_the_whole_run() -> None:
    """D6(a): a black box takes one env var at container start, not one per phase."""
    assert system_key_name("r1", "bm25") == "eval:r1:bm25:system"


def test_the_metered_harness_phases_are_the_model_calling_ones() -> None:
    assert PHASES == ("answer", "judge")


async def test_provisioning_creates_a_key_per_harness_phase() -> None:
    created: list[dict[str, Any]] = []
    async with gateway(key_handler(created)) as gw:
        keys = await gw.provision("r1", "bm25", allowed_models=["gpt-4o-mini"])
    assert set(keys.harness) == {"answer", "judge"}


async def test_provisioning_creates_one_system_key() -> None:
    created: list[dict[str, Any]] = []
    async with gateway(key_handler(created)) as gw:
        keys = await gw.provision("r1", "bm25", allowed_models=["gpt-4o-mini"])
    assert keys.system.vk_id


async def test_every_key_pins_the_allowed_models() -> None:
    """Pinning is a mechanism, not policy text: a blocked model gets a 403."""
    created: list[dict[str, Any]] = []
    async with gateway(key_handler(created)) as gw:
        await gw.provision("r1", "bm25", allowed_models=["gpt-4o-mini"])
    for body in created:
        models = body["provider_configs"][0]["allowed_models"]
        assert models == ["gpt-4o-mini"]


async def test_every_key_binds_the_providers_upstream_credentials() -> None:
    """A key that pins models but binds no credential cannot make a single call.

    A keyed request resolves against the entry's own credential list, which defaults to
    empty, so pinning a provider without binding one yields `no keys found for provider` on
    every metered call while unkeyed traffic still succeeds — a broken run rather than a
    missing number. `allowed_models` enforces the pin; this only says who pays, so `*` is
    what a benchmark wants.

    The spelling is asserted because two plausible alternatives are silently accepted and
    dropped by a live 1.6.7 gateway — `keys`, which is the name the read side reports, and
    `allow_all_keys`, the column this sets. Both provisioned keys that could not spend.
    """
    created: list[dict[str, Any]] = []
    async with gateway(key_handler(created)) as gw:
        await gw.provision("r1", "bm25", allowed_models=[PinnedModel("gpt-4o-mini", "openai")])
    for body in created:
        assert body["provider_configs"][0]["key_ids"] == ["*"]


async def test_a_pinned_model_names_the_provider_it_applies_to() -> None:
    """D11: an entry without a `provider` restricts nothing on a multi-provider gateway."""
    created: list[dict[str, Any]] = []
    async with gateway(key_handler(created)) as gw:
        await gw.provision(
            "r1",
            "bm25",
            allowed_models=[PinnedModel("gpt-4o-mini", "openai")],
        )
    for body in created:
        assert body["provider_configs"] == [
            {
                "provider": "openai",
                "allowed_models": ["gpt-4o-mini"],
                "key_ids": ["*"],
            }
        ]


async def test_models_are_grouped_by_their_provider() -> None:
    """One entry per provider, each listing only the models that provider serves."""
    created: list[dict[str, Any]] = []
    async with gateway(key_handler(created)) as gw:
        await gw.provision(
            "r1",
            "bm25",
            allowed_models=[
                PinnedModel("qwen3-14b", "vllm"),
                PinnedModel("gpt-4o-mini", "openai"),
                PinnedModel("text-embedding-3-small", "openai"),
            ],
        )
    assert created[0]["provider_configs"] == [
        {
            "provider": "openai",
            "allowed_models": ["gpt-4o-mini", "text-embedding-3-small"],
            "key_ids": ["*"],
        },
        {"provider": "vllm", "allowed_models": ["qwen3-14b"], "key_ids": ["*"]},
    ]


async def test_a_bare_model_id_still_pins_what_it_can() -> None:
    """A v1 suite names no provider, and a run under it must still be provisionable."""
    created: list[dict[str, Any]] = []
    async with gateway(key_handler(created)) as gw:
        await gw.provision("r1", "bm25", allowed_models=[PinnedModel("gpt-4o-mini")])
    assert created[0]["provider_configs"] == [{"allowed_models": ["gpt-4o-mini"], "key_ids": ["*"]}]


async def test_a_created_key_is_read_from_the_nested_envelope() -> None:
    """Bifrost nests the key under `virtual_key` and names its id `id`, not `vk_id`."""
    async with gateway(
        lambda request: httpx.Response(200, json=created_response("abc-123", "sk-bf-live"))
    ) as gw:
        keys = await gw.provision("r1", "bm25", allowed_models=["m"])
    assert (keys.system.vk_id, keys.system.value) == ("abc-123", "sk-bf-live")


async def test_a_key_response_missing_its_id_is_reported() -> None:
    """A 2xx that carries no usable id cannot be metered against, so it is not accepted."""
    async with gateway(
        lambda request: httpx.Response(200, json={"message": "ok", "virtual_key": {}})
    ) as gw:
        with pytest.raises(GatewayError, match="carries no key id"):
            await gw.provision("r1", "bm25", allowed_models=["m"])


async def test_a_failed_key_creation_is_reported() -> None:
    async with gateway(lambda request: httpx.Response(500, text="boom")) as gw:
        with pytest.raises(GatewayError, match="virtual key"):
            await gw.provision("r1", "bm25", allowed_models=["m"])


async def test_a_scrape_returns_usage_per_key() -> None:
    async with gateway(key_handler([])) as gw:
        snapshot = await gw.snapshot()
    assert snapshot["vk_1"] == Usage(prompt_tokens=50)


async def test_a_failed_scrape_is_reported() -> None:
    async with gateway(lambda request: httpx.Response(503, text="unavailable")) as gw:
        with pytest.raises(GatewayError, match="metrics"):
            await gw.snapshot()


def test_request_headers_carry_the_key_and_the_run_dimensions() -> None:
    """`x-bf-dim-*` keeps Prometheus queries legible once many runs are in flight."""
    headers = Gateway.headers_for("sk-abc", run_id="r1", phase="answer")
    assert headers["x-bf-vk"] == "sk-abc"
    assert headers["x-bf-dim-run"] == "r1"
    assert headers["x-bf-dim-phase"] == "answer"
