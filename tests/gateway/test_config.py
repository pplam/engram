"""The gateway config is data and must stay consistent with the suite (§3.3).

These are cheap structural checks, not a substitute for verifying the fields against a
running image. They catch the failure that costs a whole run: a model the suite pins
that the gateway was never told about.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from orchestrator.compose import ENV_REF
from orchestrator.gateway import METRICS_PATH
from orchestrator.suite import LEGACY, load_suite

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "gateway" / "bifrost.json"
EXAMPLE = REPO / "gateway" / "bifrost.example.json"
COMPOSE = REPO / "gateway" / "docker-compose.yml"

# Where the image's entrypoint looks: `$APP_DIR/config.json`, with APP_DIR defaulting to
# /app/data. Not configurable via any flag the entrypoint forwards.
APP_CONFIG_PATH = "/app/data/config.json"


@pytest.fixture
def config() -> dict[str, object]:
    """The operator's real `bifrost.json`, which is gitignored and holds live keys.

    Absent on a fresh clone, so every check over it skips rather than fails: it names
    private endpoints and cannot be committed. `bifrost.example.json` is the tracked
    template, and `example_config` below covers what can be asserted without secrets.
    """
    if not CONFIG.exists():
        pytest.skip(f"{CONFIG.name} is gitignored; copy {EXAMPLE.name} and add providers")
    raw: dict[str, object] = json.loads(CONFIG.read_text())
    return raw


@pytest.fixture
def example_config() -> dict[str, object]:
    raw: dict[str, object] = json.loads(EXAMPLE.read_text())
    return raw


def test_governance_and_virtual_keys_are_enabled(config: dict[str, object]) -> None:
    """Without governance there are no virtual keys, and no attribution at all."""
    governance = config["governance"]
    assert isinstance(governance, dict)
    assert governance["enable_virtual_keys"] is True


def test_prometheus_is_exposed_where_the_orchestrator_scrapes(config: dict[str, object]) -> None:
    telemetry = config["telemetry"]
    assert isinstance(telemetry, dict)
    assert telemetry["prometheus_path"] == METRICS_PATH


def test_the_run_and_phase_dimensions_are_declared(config: dict[str, object]) -> None:
    """`x-bf-dim-run` / `x-bf-dim-phase` keep queries legible with many runs in flight."""
    telemetry = config["telemetry"]
    assert isinstance(telemetry, dict)
    dimensions = telemetry["custom_dimensions"]
    assert isinstance(dimensions, list)
    assert set(dimensions) == {"run", "phase"}


# Every suite file in the repo, so a new pin is checked against the gateway by existing
# tests rather than by remembering to extend a list here.
VERSIONS = sorted(path.stem for path in (REPO / "suites").glob("*.yaml"))
CURRENT_SHAPE = [v for v in VERSIONS if v not in LEGACY]


@pytest.mark.parametrize("version", VERSIONS)
def test_every_model_the_suite_pins_is_available(config: dict[str, object], version: str) -> None:
    """A suite model the gateway cannot serve fails the run at the first call."""
    suite = load_suite(version, REPO)
    providers = config["providers"]
    assert isinstance(providers, dict)
    served: set[str] = set()
    for provider in providers.values():
        for key in provider["keys"]:
            served.update(key["models"])

    assert {suite.models.chat.id, suite.models.embedding.id, suite.models.judge.id} <= served


@pytest.mark.parametrize("version", CURRENT_SHAPE)
def test_every_provider_a_suite_names_has_a_block(config: dict[str, object], version: str) -> None:
    """D11: a pinned `(provider, id)` whose provider does not exist routes nowhere."""
    suite = load_suite(version, REPO)
    providers = config["providers"]
    assert isinstance(providers, dict)
    named = {
        suite.models.chat.provider,
        suite.models.embedding.provider,
        suite.models.judge.provider,
    }
    assert named <= set(providers)


@pytest.mark.parametrize("version", CURRENT_SHAPE)
def test_each_provider_serves_the_models_pinned_against_it(
    config: dict[str, object], version: str
) -> None:
    """Pairing matters, not just membership: the right id under the wrong provider 404s."""
    suite = load_suite(version, REPO)
    providers = config["providers"]
    assert isinstance(providers, dict)
    for model in (suite.models.chat, suite.models.embedding, suite.models.judge):
        assert model.provider is not None
        block = providers[model.provider]
        served = {name for key in block["keys"] for name in key["models"]}
        assert model.id in served, f"provider {model.provider!r} does not serve {model.id!r}"


@pytest.mark.parametrize("version", CURRENT_SHAPE)
def test_the_judge_is_not_the_answer_model(version: str) -> None:
    """ARCH §14: a judge that *is* the answer model grades its own output."""
    suite = load_suite(version, REPO)
    assert suite.models.judge.id != suite.models.chat.id


def test_no_base_url_carries_a_version_prefix(config: dict[str, object]) -> None:
    """Bifrost appends `/v1` itself, so a `/v1` in `base_url` sends calls to `/v1/v1/...`.

    Verified against the vLLM provider guide: the base URL points at the service root.
    The failure is a 404 on the first model call, long after the run has been paid for.
    """
    providers = config["providers"]
    assert isinstance(providers, dict)
    for name, provider in providers.items():
        base_url = (provider.get("network_config") or {}).get("base_url")
        if not isinstance(base_url, str):
            continue
        assert not base_url.rstrip("/").endswith("/v1"), (
            f"provider {name!r} sets base_url to {base_url!r}; Bifrost appends the "
            "version prefix, so this resolves to /v1/v1"
        )


def test_every_dataset_pinned_judge_is_routable(config: dict[str, object]) -> None:
    """LoCoMo-Refined's qwen3-14b is why a provider exists at all; it must resolve."""
    providers = config["providers"]
    assert isinstance(providers, dict)
    for path in sorted((REPO / "datasets").glob("*/dataset.yaml")):
        pinned = (yaml.safe_load(path.read_text()) or {}).get("official_judge")
        if not pinned:
            continue
        provider = pinned.get("provider")
        assert provider in providers, f"{path.parent.name} pins provider {provider!r}"
        served = {name for key in providers[provider]["keys"] for name in key["models"]}
        assert pinned["id"] in served


def test_base_url_is_never_an_env_reference(config: dict[str, object]) -> None:
    """`network_config.base_url` is a plain string in Bifrost, not a SecretVar (D11).

    An `env.*` value there is used literally, so the request silently goes nowhere.
    Only SecretVar fields — `keys[].value`, `vllm_key_config.url` — resolve `env.*`.
    """
    providers = config["providers"]
    assert isinstance(providers, dict)
    for name, provider in providers.items():
        network = provider.get("network_config") or {}
        base_url = network.get("base_url")
        assert not (isinstance(base_url, str) and base_url.startswith("env.")), (
            f"provider {name!r} sets base_url to {base_url!r}; that field takes a literal "
            "URL, so an env reference would be sent verbatim"
        )


def test_no_credential_is_written_into_the_config(config: dict[str, object]) -> None:
    """Keys are referenced by env var name; a literal key in a local file is a hazard.

    Takes the fixture so it skips with the rest when the file is absent. This one is
    advisory now that the file is gitignored — `test_every_key_in_the_example_is_an_env_reference`
    is the check that guards what actually ships.
    """
    assert "sk-" not in CONFIG.read_text()


def test_the_example_config_is_tracked_and_the_real_one_is_not() -> None:
    """The template ships; the operator's copy holds live keys and private hosts."""
    assert EXAMPLE.exists()
    ignored = subprocess.run(
        ["git", "check-ignore", "gateway/bifrost.json", "gateway/bifrost.example.json"],
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert "gateway/bifrost.json" in ignored
    assert "gateway/bifrost.example.json" not in ignored


def test_every_key_in_the_example_is_an_env_reference(example_config: dict[str, object]) -> None:
    """A literal credential in the tracked template is the leak this split exists to stop.

    Stronger than a `sk-` substring check: every `keys[].value` must be an `env.*` ref, so
    a key in any provider's format fails rather than only the OpenAI-shaped ones.
    """
    providers = example_config["providers"]
    assert isinstance(providers, dict)
    for name, provider in providers.items():
        for key in provider["keys"]:
            assert ENV_REF.match(key["value"]), (
                f"provider {name!r} inlines a credential; use env.NAME so the value stays "
                "in the environment"
            )


def test_the_example_declares_no_private_endpoint(example_config: dict[str, object]) -> None:
    """`proxy` is a private deployment: its host does not belong in a public template."""
    providers = example_config["providers"]
    assert isinstance(providers, dict)
    assert "proxy" not in providers
    assert "proxy" not in EXAMPLE.read_text()


def test_the_example_serves_the_providers_the_open_suites_pin(
    example_config: dict[str, object],
) -> None:
    """The template is a working starting point, not just a syntax demo.

    `v2` pins every model against `openai`, and LoCoMo-Refined's `official_judge` needs
    `vllm`, so those two blocks make the example runnable as shipped.
    """
    providers = example_config["providers"]
    assert isinstance(providers, dict)
    assert {"openai", "vllm"} <= set(providers)


@pytest.mark.parametrize("field", ["governance", "telemetry", "client"])
def test_the_example_carries_the_same_platform_settings(
    example_config: dict[str, object], field: str
) -> None:
    """Copying the template must not silently lose virtual keys or metrics attribution."""
    assert field in example_config


def test_the_example_enables_virtual_keys_and_prometheus(
    example_config: dict[str, object],
) -> None:
    governance = example_config["governance"]
    telemetry = example_config["telemetry"]
    assert isinstance(governance, dict)
    assert isinstance(telemetry, dict)
    assert governance["enable_virtual_keys"] is True
    assert telemetry["prometheus_path"] == METRICS_PATH


def test_no_base_url_in_the_example_is_malformed(example_config: dict[str, object]) -> None:
    """Same two D11 traps as the real config: no `/v1` suffix, no `env.*` in `base_url`."""
    providers = example_config["providers"]
    assert isinstance(providers, dict)
    for name, provider in providers.items():
        base_url = (provider.get("network_config") or {}).get("base_url")
        if not isinstance(base_url, str):
            continue
        assert not base_url.rstrip("/").endswith("/v1"), f"provider {name!r} doubles /v1"
        assert not base_url.startswith("env."), f"provider {name!r} env-refs a plain field"


def test_the_system_network_has_no_route_off_the_host() -> None:
    """Egress policy: a system reaches the gateway and nothing else (§3.3)."""
    compose = yaml.safe_load(COMPOSE.read_text())
    assert compose["networks"]["models"]["internal"] is True


def test_only_the_gateway_bridges_to_egress() -> None:
    compose = yaml.safe_load(COMPOSE.read_text())
    on_egress = [
        name
        for name, service in compose["services"].items()
        if "egress" in (service.get("networks") or [])
    ]
    assert on_egress == ["gateway"]


def test_the_system_under_test_is_pointed_at_the_gateway() -> None:
    compose = yaml.safe_load(COMPOSE.read_text())
    env = compose["services"]["system_under_test"]["environment"]
    assert any("MODEL_BASE_URL=http://gateway:8080" in entry for entry in env)


def test_the_config_is_mounted_where_bifrost_reads_it() -> None:
    """Bifrost reads `$APP_DIR/config.json`; it has no flag for a config path.

    The entrypoint's last line is `exec /app/main -app-dir ... -port ... -host ...` — no
    config argument exists to pass. A config mounted anywhere else is silently ignored and
    the gateway boots with zero providers.
    """
    compose = yaml.safe_load(COMPOSE.read_text())
    volumes = compose["services"]["gateway"]["volumes"]
    targets = [entry.split(":")[1] for entry in volumes]
    assert APP_CONFIG_PATH in targets, (
        f"gateway mounts {targets}; Bifrost only reads its config from {APP_CONFIG_PATH}"
    )


def test_no_config_flag_is_passed_to_the_entrypoint() -> None:
    """An unrecognized argument makes the entrypoint spin forever without starting.

    `parse_args` falls through to `set -- "$@" "$1"; shift`, which re-appends the unmatched
    argument to the tail, so `$#` never reaches 0 and `exec /app/main` is never reached.
    The container reports `Up`, logs nothing, pegs a core, and listens on no port —
    verified against v1.6.7. Its `if [ $# -gt 1 ]` guard means two args are enough to hit
    it, which is exactly what a `-config <path>` pair is.
    """
    compose = yaml.safe_load(COMPOSE.read_text())
    command = compose["services"]["gateway"].get("command")
    assert not command, (
        f"gateway passes {command!r}; the entrypoint loops forever on any argument it does "
        "not recognize, and it recognizes only --port/--host"
    )


def test_no_required_variable_guard_blocks_a_gateway_only_start() -> None:
    """`docker compose` interpolates the whole file even when one service is named.

    So a `${VAR:?}` guard anywhere — including on the system-under-test template — aborts
    `bench gateway` for a variable the gateway itself never needs. Verified by running it:
    `up -d gateway` failed on SUT_IMAGE. Defaults keep the template documentary without
    making it a precondition; a system actually being benchmarked overrides them.
    """
    # Matched inside a `${...}` interpolation only: prose explaining the rule is allowed
    # to name the syntax it forbids.
    guards = re.findall(r"\$\{[A-Za-z_][A-Za-z0-9_]*:\?[^}]*\}", COMPOSE.read_text())
    assert guards == [], (
        f"{guards} abort interpolation for every service, so `up -d gateway` fails on a "
        "variable only the system-under-test template uses"
    )
