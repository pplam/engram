"""Virtual-key provisioning and metering against Bifrost (§3.1, §3.2, D3, D5, D6).

Two key families per run, because the two sides differ in what they can hold:

- **Harness** — one key per `(run_id, system, phase)` (D3). We make those calls, so
  per-phase attribution is exact.
- **System under test** — one key per `(run_id, system)` (D6(a)). A black box gets one
  env var at container start and cannot be handed a phase-scoped key, so its per-phase
  split comes from the `/metrics` snapshot boundary instead.

That difference is why the report labels harness cost `measured` and system cost
`derived`: collapsing them would overstate precision on the half that is inferred.

`allowed_models` is the pinning mechanism — a non-pinned model gets a 403 from the
gateway rather than relying on anyone honouring policy text. Each entry names the
provider it applies to (D11): on a multi-provider gateway a bare model list restricts
nothing, so a key posted without a provider was pinning in name only.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import httpx

from orchestrator.metrics import MetricsError, Snapshot, parse_metrics

# The harness phases that call a model. `ingest` and `retrieve` are the system's own
# traffic, and the other stages make no model calls at all.
PHASES: tuple[str, ...] = ("answer", "judge")

KEYS_PATH = "/api/governance/virtual-keys"
METRICS_PATH = "/metrics"


class GatewayError(Exception):
    """The gateway refused a request, or returned something unusable."""


@dataclass(frozen=True)
class PinnedModel:
    """One model a run may reach, and the provider that serves it (D11).

    `provider` is None only for a legacy suite that names bare ids. Those keys pin the
    model list and nothing else, which is all the suite said.
    """

    id: str
    provider: str | None = None


def wire_name(model: PinnedModel) -> str:
    """Return how `model` is named in a request body: `<provider>/<id>`.

    Bifrost reads the segment before the first `/` as the provider, so an id that itself
    contains a slash — `deepseek/deepseek-v4-flash-0731` — is read as provider `deepseek`
    and fails with `failed to get config for provider deepseek`. Prefixing the provider is
    what makes the pinned `(provider, id)` pair addressable (D11). Verified against a
    running gateway, not inferred.

    A legacy model with no provider keeps its bare id: v1 predates providers, and its rows
    are reproduced with the name they were published under.
    """
    return f"{model.provider}/{model.id}" if model.provider else model.id


def _provider_configs(models: Sequence[PinnedModel | str]) -> list[dict[str, Any]]:
    """Group `models` into one `provider_configs` entry per provider, sorted."""
    grouped: dict[str | None, list[str]] = {}
    for model in models:
        pinned = model if isinstance(model, PinnedModel) else PinnedModel(model)
        grouped.setdefault(pinned.provider, []).append(pinned.id)

    configs: list[dict[str, Any]] = []
    # `None` sorts last under a str key, so legacy entries never displace a real
    # provider's ordering. Sorting at all is what keeps a provisioning payload
    # comparable between runs.
    for provider in sorted(grouped, key=lambda p: (p is None, p or "")):
        entry: dict[str, Any] = {}
        if provider is not None:
            entry["provider"] = provider
        entry["allowed_models"] = sorted(grouped[provider])
        # Which upstream credential the call is charged to. Without this the entry binds
        # none, and every metered call fails with `no keys found for provider` while
        # unkeyed traffic still succeeds — so the symptom is a broken run, not a missing
        # number. `*` is all of that provider's keys, which is what a benchmark wants:
        # `allowed_models` above enforces the pin, and this only says who pays.
        #
        # Verified against a live 1.6.7 gateway, because two plausible spellings are
        # silently ignored rather than refused: `keys` (the name the read side reports)
        # and `allow_all_keys` (the column this sets, accepted and dropped on both POST
        # and PUT). A wrong spelling here provisions keys that cannot make one call.
        entry["key_ids"] = ["*"]
        configs.append(entry)
    return configs


@dataclass(frozen=True)
class VirtualKey:
    """One provisioned key: its id labels the metrics, its value authorizes calls."""

    name: str
    vk_id: str
    value: str


@dataclass(frozen=True)
class RunKeys:
    """The keys for one run: phase-scoped for the harness, run-scoped for the system."""

    harness: dict[str, VirtualKey]
    system: VirtualKey


def harness_key_name(run_id: str, system: str, phase: str) -> str:
    """Return the virtual-key name for one harness phase (D3)."""
    return f"eval:{run_id}:{system}:{phase}"


def system_key_name(run_id: str, system: str) -> str:
    """Return the single virtual-key name a system under test gets for a run (D6(a))."""
    return f"eval:{run_id}:{system}:system"


def _created_key(body: Any) -> tuple[str, str] | None:
    """Return `(vk_id, value)` from a key-creation response, or None if it carries neither.

    Bifrost nests the key under `virtual_key` and calls its identifier `id`; the API docs
    describe a flat `vk_id`. Both are read because the docs' shape is what this code was
    written against and some version may yet send it, but the live shape is checked first.
    The id is what `/metrics` labels traffic with, so a response without one cannot be
    metered against and is treated as a failure rather than a keyless success.
    """
    if not isinstance(body, dict):
        return None
    nested = body.get("virtual_key")
    record: Any = nested if isinstance(nested, dict) else body
    vk_id = record.get("id") or record.get("vk_id")
    if not vk_id:
        return None
    return str(vk_id), str(record.get("value") or "")


class Gateway:
    """Provisions run keys and reads the counters that per-phase cost is derived from."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._client.aclose()

    @staticmethod
    def headers_for(value: str, run_id: str, phase: str) -> dict[str, str]:
        """Return the headers that carry a key and label its traffic in Prometheus."""
        return {
            "x-bf-vk": value,
            "x-bf-dim-run": run_id,
            "x-bf-dim-phase": phase,
        }

    async def _create_key(
        self, name: str, allowed_models: Sequence[PinnedModel | str]
    ) -> VirtualKey:
        payload: dict[str, Any] = {
            "name": name,
            "provider_configs": _provider_configs(allowed_models),
        }
        try:
            response = await self._client.post(KEYS_PATH, json=payload)
        except httpx.HTTPError as err:
            raise GatewayError(f"cannot reach the gateway to create virtual key {name!r}") from err

        if response.status_code // 100 != 2:
            raise GatewayError(
                f"gateway refused virtual key {name!r} with HTTP {response.status_code}"
            )

        try:
            body: Any = response.json()
        except ValueError as err:
            raise GatewayError(f"virtual key {name!r} response is not JSON") from err
        key = _created_key(body)
        if key is None:
            raise GatewayError(f"virtual key {name!r} response carries no key id")

        return VirtualKey(name=name, vk_id=key[0], value=key[1])

    async def provision(
        self, run_id: str, system: str, allowed_models: Sequence[PinnedModel | str]
    ) -> RunKeys:
        """Create this run's harness and system keys, pinned to `allowed_models`."""
        harness = {
            phase: await self._create_key(harness_key_name(run_id, system, phase), allowed_models)
            for phase in PHASES
        }
        return RunKeys(
            harness=harness,
            system=await self._create_key(system_key_name(run_id, system), allowed_models),
        )

    async def snapshot(self) -> Snapshot:
        """Scrape `/metrics` and return cumulative usage per virtual key."""
        try:
            response = await self._client.get(METRICS_PATH)
        except httpx.HTTPError as err:
            raise GatewayError("cannot reach the gateway metrics endpoint") from err
        if response.status_code // 100 != 2:
            raise GatewayError(f"gateway metrics returned HTTP {response.status_code}")
        try:
            return parse_metrics(response.text)
        except MetricsError as err:
            raise GatewayError(f"gateway metrics are unreadable: {err}") from err
