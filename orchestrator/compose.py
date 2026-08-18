"""Bringing the model plane up and down (§3.3).

`bench gateway start` is a thin wrapper over `docker compose`: the compose file and
`bifrost.json` stay the source of truth, and nothing here duplicates their contents. What
it adds is the two things a user gets wrong by hand — starting the system-under-test
template alongside the gateway, and running a benchmark against a gateway that has not
finished booting.

Credentials are checked by *name* only. `bifrost.json` references them as `env.NAME`, so
the missing ones can be named in an error message without any value being read or logged.
"""

import asyncio
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import httpx

from orchestrator.gateway import METRICS_PATH

# The compose file also defines a digest-pinned system-under-test template. It belongs to
# whoever is benchmarking a container, so `up` names the gateway rather than defaulting to
# every service in the file.
GATEWAY_SERVICE = "gateway"

DEFAULT_COMPOSE = Path("gateway/docker-compose.yml")
DEFAULT_CONFIG = Path("gateway/bifrost.json")
DEFAULT_URL = "http://127.0.0.1:8080"

# A `SecretVar` in Bifrost's config: `env.NAME` resolves to that environment variable.
ENV_REF = re.compile(r"^env\.([A-Za-z_][A-Za-z0-9_]*)$")


class ComposeError(Exception):
    """The gateway could not be started, or its config is unusable."""


def up_argv(compose_file: Path, service: str = GATEWAY_SERVICE) -> list[str]:
    """Return the `docker compose` argv that starts one service detached."""
    return ["docker", "compose", "-f", str(compose_file), "up", "-d", service]


def stop_argv(compose_file: Path, service: str = GATEWAY_SERVICE) -> list[str]:
    """Return the `docker compose` argv that stops one service.

    `stop` rather than `down`: it leaves the container in place so a later `start` brings
    the same one back, and it removes nothing. Named per service for the same reason `up`
    is — a system under test defined in the same file keeps running.
    """
    return ["docker", "compose", "-f", str(compose_file), "stop", service]


def _walk(node: Any) -> Iterable[tuple[str, Any]]:
    """Yield every `(key, value)` pair in a nested JSON document."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key), value
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def credential_vars(config_file: Path) -> list[str]:
    """Return the env vars `bifrost.json` references, in first-seen order.

    Only exact `env.NAME` values count. The config documents itself in `_comment` blocks
    that mention variables in prose, and a substring match there would invent a
    requirement the gateway does not actually have.
    """
    try:
        config = json.loads(config_file.read_text())
    except (OSError, ValueError) as err:
        raise ComposeError(f"cannot read {config_file}: {err}") from err

    names: list[str] = []
    for key, value in _walk(config):
        if key.startswith("_comment") or not isinstance(value, str):
            continue
        match = ENV_REF.match(value)
        if match and match.group(1) not in names:
            names.append(match.group(1))
    return names


def unset_credentials(names: Iterable[str], environ: Mapping[str, str]) -> list[str]:
    """Return the names among `names` that are absent or empty in `environ`."""
    return [name for name in names if not environ.get(name)]


async def is_ready(client: httpx.AsyncClient) -> bool:
    """Return whether the gateway answers `/metrics` right now.

    One request, no retry: the caller decides whether an unready gateway means "still
    booting" (`wait_ready`) or "report it down" (`bench gateway status`).
    """
    try:
        response = await client.get(METRICS_PATH)
    except httpx.HTTPError:
        # Nothing listening on the port, or the connection died mid-request.
        return False
    return response.status_code // 100 == 2


async def wait_ready(
    client: httpx.AsyncClient, timeout_s: float = 60.0, interval: float = 0.5
) -> None:
    """Poll `/metrics` until the gateway answers, or raise once `timeout_s` elapses.

    `/metrics` rather than a dedicated health path: it is the endpoint per-phase cost is
    derived from, so a gateway that answers it is ready for the thing we actually need.
    """
    try:
        async with asyncio.timeout(timeout_s):
            while True:
                if await is_ready(client):
                    return
                await asyncio.sleep(interval)
    except TimeoutError as err:
        raise ComposeError(
            f"gateway not ready after {timeout_s:g}s: {client.base_url}{METRICS_PATH} "
            "did not answer"
        ) from err
