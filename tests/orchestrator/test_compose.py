"""Launching the model plane is argv construction plus a readiness poll (§3.3).

Nothing here starts a container or opens a socket: `up_argv` is pure, and `wait_ready`
takes the client so a `MockTransport` stands in for the gateway.
"""

import json
from pathlib import Path

import httpx
import pytest

from orchestrator.compose import (
    GATEWAY_SERVICE,
    ComposeError,
    credential_vars,
    is_ready,
    stop_argv,
    unset_credentials,
    up_argv,
    wait_ready,
)


def test_up_argv_starts_only_the_gateway_service(tmp_path: Path) -> None:
    """The compose file also defines a system-under-test template; `up` must skip it."""
    compose = tmp_path / "docker-compose.yml"
    assert up_argv(compose) == [
        "docker",
        "compose",
        "-f",
        str(compose),
        "up",
        "-d",
        GATEWAY_SERVICE,
    ]


def test_stop_argv_stops_only_the_gateway_service(tmp_path: Path) -> None:
    """Mirror of `up_argv`: a benchmarked container in the same file keeps running."""
    compose = tmp_path / "docker-compose.yml"
    assert stop_argv(compose) == [
        "docker",
        "compose",
        "-f",
        str(compose),
        "stop",
        GATEWAY_SERVICE,
    ]


def test_credential_vars_collects_env_references(tmp_path: Path) -> None:
    config = tmp_path / "bifrost.json"
    config.write_text(
        json.dumps(
            {
                "providers": {
                    "openai": {"keys": [{"value": "env.OPENAI_API_KEY"}]},
                    "vllm": {
                        "keys": [
                            {
                                "value": "env.VLLM_API_KEY",
                                "vllm_key_config": {"url": "env.VLLM_BASE_URL"},
                            }
                        ]
                    },
                }
            }
        )
    )
    assert credential_vars(config) == ["OPENAI_API_KEY", "VLLM_API_KEY", "VLLM_BASE_URL"]


def test_credential_vars_ignores_prose_in_comments(tmp_path: Path) -> None:
    """`bifrost.json` documents itself in `_comment` blocks that name env vars in prose."""
    config = tmp_path / "bifrost.json"
    config.write_text(
        json.dumps(
            {
                "_comment": ["env.VLLM_API_KEY is read from the environment"],
                "providers": {"openai": {"keys": [{"value": "env.OPENAI_API_KEY"}]}},
            }
        )
    )
    assert credential_vars(config) == ["OPENAI_API_KEY"]


def test_credential_vars_rejects_unreadable_config(tmp_path: Path) -> None:
    config = tmp_path / "bifrost.json"
    config.write_text("{not json")
    with pytest.raises(ComposeError, match="bifrost.json"):
        credential_vars(config)


def test_unset_credentials_names_only_the_missing_ones() -> None:
    environ = {"OPENAI_API_KEY": "sk-live", "VLLM_API_KEY": ""}
    assert unset_credentials(["OPENAI_API_KEY", "VLLM_API_KEY", "OTHER"], environ) == [
        "VLLM_API_KEY",
        "OTHER",
    ]


def _client(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        base_url="http://gw.invalid",
    )


async def test_wait_ready_returns_once_metrics_answers() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, text="")

    async with _client(handler) as client:
        await wait_ready(client, timeout_s=1.0, interval=0.0)
    assert seen == ["/metrics"]


async def test_wait_ready_retries_until_the_gateway_comes_up() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, text="")

    async with _client(handler) as client:
        await wait_ready(client, timeout_s=5.0, interval=0.0)
    assert attempts == 3


async def test_wait_ready_gives_up_after_the_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="")

    async with _client(handler) as client:
        with pytest.raises(ComposeError, match="not ready"):
            await wait_ready(client, timeout_s=0.0, interval=0.0)


async def test_is_ready_is_true_when_metrics_answers() -> None:
    """The same endpoint `wait_ready` polls, asked once rather than until a deadline."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, text="")

    async with _client(handler) as client:
        assert await is_ready(client) is True
    assert seen == ["/metrics"]


async def test_is_ready_is_false_when_nothing_is_listening() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with _client(handler) as client:
        assert await is_ready(client) is False


async def test_is_ready_is_false_when_the_gateway_answers_an_error() -> None:
    """Up but not serving `/metrics` is not usable: cost is derived from that endpoint."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="")

    async with _client(handler) as client:
        assert await is_ready(client) is False
