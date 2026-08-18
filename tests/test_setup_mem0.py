"""`scripts/setup_mem0.py` configures the mem0 server to run against the gateway.

The pins live in `suites/*.yaml`, but mem0 reads its LLM and embedder config from its own
`.env` and `POST /configure` — outside this repo, so nothing in `orchestrator/` can state
or check them (see `registry/mem0.yaml`). This script is the one place that
translates a suite into that config, which makes drift between the two the thing worth
testing: a model id, a dimension, or a temperature that disagrees with the suite produces
a row measured under a pin nobody declared.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from orchestrator.suite import CURRENT_SUITE, Suite, load_suite
from scripts.setup_mem0 import (
    DEFAULT_SERVER,
    SetupError,
    collection_name,
    configure_payload,
    dead_services,
    env_file,
    override_yaml,
    postgres_password,
    reset_argv,
    reset_paths,
    summary,
)

REPO = Path(__file__).resolve().parents[1]
GATEWAY_URL = "http://gateway:8080/v1"
BASE_COMPOSE = REPO / DEFAULT_SERVER / "docker-compose.yaml"


def compose(rendered: str) -> Any:
    """Return the override parsed the way `docker compose` would read it."""
    return yaml.safe_load(rendered)


@pytest.fixture
def suite() -> Suite:
    return load_suite(CURRENT_SUITE, REPO)


def test_env_points_both_halves_at_the_gateway(suite: Suite) -> None:
    """One variable pins llm and embedder alike: both read OPENAI_BASE_URL."""
    rendered = env_file(suite, key="vk-x", password="pw", gateway_url=GATEWAY_URL)
    assert f"OPENAI_BASE_URL={GATEWAY_URL}" in rendered


def test_env_names_models_with_their_provider_prefix(suite: Suite) -> None:
    """Bifrost reads the segment before the first `/` as the provider (gateway.wire_name).

    A bare `deepseek/deepseek-v4-flash-0731` resolves to provider `deepseek`, which the
    gateway does not serve.
    """
    rendered = env_file(suite, key="vk-x", password="pw", gateway_url=GATEWAY_URL)
    assert "MEM0_DEFAULT_LLM_MODEL=proxy/deepseek/deepseek-v4-flash-0731" in rendered
    assert "MEM0_DEFAULT_EMBEDDER_MODEL=proxy/qwen/qwen3-embedding-8b" in rendered


def test_env_carries_a_virtual_key_when_one_is_given(suite: Suite) -> None:
    """A gateway with governance on still needs one, and it goes here, not in the URL."""
    rendered = env_file(suite, key="vk-secret", password="pw", gateway_url=GATEWAY_URL)
    assert "OPENAI_API_KEY=vk-secret" in rendered


def test_env_still_sets_a_credential_when_the_gateway_wants_none(suite: Suite) -> None:
    """The value stops being a credential but cannot become absent.

    mem0 builds an `openai.OpenAI` client, which raises `OpenAIError` at construction
    when no key is present anywhere (`_client.py`). A placeholder is what keeps a keyless
    gateway from failing before the first request is made.
    """
    rendered = env_file(suite, key=None, password="pw", gateway_url=GATEWAY_URL)
    line = next(ln for ln in rendered.splitlines() if ln.startswith("OPENAI_API_KEY="))
    value = line.split("=", 1)[1].strip()
    assert value and value != "None", "a stringified None is not a deliberate placeholder"


def test_env_disables_telemetry(suite: Suite) -> None:
    """mem0 phones home by default, and the internal network has nowhere for it to go."""
    rendered = env_file(suite, key="vk-x", password="pw", gateway_url=GATEWAY_URL)
    assert "MEM0_TELEMETRY=false" in rendered


def test_env_disables_auth_so_configure_is_reachable(suite: Suite) -> None:
    """`POST /configure` is `require_admin`; AUTH_DISABLED=true is what admits it."""
    rendered = env_file(suite, key="vk-x", password="pw", gateway_url=GATEWAY_URL)
    assert "AUTH_DISABLED=true" in rendered


def test_configure_sets_the_dimension_on_both_the_embedder_and_the_store(suite: Suite) -> None:
    """pgvector writes `vector(N)` at CREATE TABLE and never widens it afterwards.

    mem0 defaults to 1536 while suite v3 pins 4096, so a store left at the default
    rejects every insert on a dimension mismatch.
    """
    payload = configure_payload(suite, gateway_url=GATEWAY_URL)
    assert payload["embedder"]["config"]["embedding_dims"] == 4096
    assert payload["vector_store"]["config"]["embedding_model_dims"] == 4096


def test_configure_turns_off_the_ann_index_that_cannot_hold_the_suites_width(
    suite: Suite,
) -> None:
    """pgvector's hnsw index caps at 2000 dimensions; suite v3 pins 4096.

    mem0 defaults `hnsw` to True and, unlike its `diskann` branch, guards it with no
    dimension check — so `CREATE INDEX` raises ProgramLimitExceeded and every `add`
    returns 502. Off means exact search: slower, and the only option at this width.
    """
    config = configure_payload(suite, gateway_url=GATEWAY_URL)["vector_store"]["config"]
    assert config["hnsw"] is False
    assert config["diskann"] is False


def test_configure_pins_the_temperature_the_suite_declares(suite: Suite) -> None:
    """mem0's DEFAULT_CONFIG hardcodes 0.2 where every suite since v1 pins 0."""
    payload = configure_payload(suite, gateway_url=GATEWAY_URL)
    assert payload["llm"]["config"]["temperature"] == 0


def test_configure_pins_the_completion_budget_the_suite_declares(suite: Suite) -> None:
    """mem0 defaults to 2000, which a reasoning model spends before emitting content.

    The extraction call then returns an empty completion, mem0 parses it as "no facts",
    and the chunk is not stored while `add` still answers HTTP 200 — an ingest that
    reports success against a store that received nothing.
    """
    payload = configure_payload(suite, gateway_url=GATEWAY_URL)
    assert payload["llm"]["config"]["max_tokens"] == suite.models.chat.max_tokens


def test_configure_omits_the_budget_when_the_suite_pins_none(suite: Suite) -> None:
    """v1..v3 pin none; sending null would override mem0's default with nothing."""
    unpinned = suite.models.chat.model_copy(update={"max_tokens": None})
    without = suite.model_copy(
        update={"models": suite.models.model_copy(update={"chat": unpinned})}
    )
    assert "max_tokens" not in configure_payload(without, gateway_url=GATEWAY_URL)["llm"]["config"]


def test_configure_names_models_by_wire_name(suite: Suite) -> None:
    payload = configure_payload(suite, gateway_url=GATEWAY_URL)
    assert payload["llm"]["config"]["model"] == "proxy/deepseek/deepseek-v4-flash-0731"
    assert payload["embedder"]["config"]["model"] == "proxy/qwen/qwen3-embedding-8b"


def test_configure_refuses_a_suite_that_pins_no_dimension(suite: Suite) -> None:
    """Guessing the width would create a column the embedder cannot write into.

    Every committed suite pins one, so this suite is synthetic — the guard is for a future
    suite that omits it, where the failure would otherwise be a rejected insert per chunk.
    """
    unpinned = suite.models.embedding.model_copy(update={"dimensions": None})
    without = suite.model_copy(
        update={"models": suite.models.model_copy(update={"embedding": unpinned})}
    )
    with pytest.raises(SetupError, match="dimensions"):
        configure_payload(without, gateway_url=GATEWAY_URL)


def test_the_collection_is_scoped_to_the_suite(suite: Suite) -> None:
    """A new suite may change the embedder, and an existing table keeps its old width."""
    assert collection_name(suite) == f"engram_{CURRENT_SUITE}"


def test_an_existing_password_is_reused_rather_than_rotated(tmp_path: Path) -> None:
    """The Postgres volume outlives the .env; a fresh password would lock us out of it."""
    existing = tmp_path / ".env"
    existing.write_text("POSTGRES_PASSWORD=original\nAUTH_DISABLED=true\n")
    assert postgres_password(existing, from_env=None) == "original"


def test_a_password_is_generated_when_there_is_nothing_to_reuse(tmp_path: Path) -> None:
    generated = postgres_password(tmp_path / "absent.env", from_env=None)
    assert len(generated) >= 16


def test_an_explicit_password_wins(tmp_path: Path) -> None:
    existing = tmp_path / ".env"
    existing.write_text("POSTGRES_PASSWORD=original\n")
    assert postgres_password(existing, from_env="chosen") == "chosen"


def test_the_override_attaches_mem0_to_the_gateway_network() -> None:
    """The gateway is another compose project, so its network joins as external."""
    rendered = override_yaml("gateway_models")
    assert "gateway_models" in rendered
    assert "external: true" in rendered


def test_the_override_leaves_mem0s_own_network_routable() -> None:
    """Published ports and `internal: true` are mutually exclusive, and the ports win.

    Docker cannot publish a port for a container whose every network is internal: the
    host got a Docker Desktop 502 and `docker port` listed nothing. Since the harness
    dials 127.0.0.1:8888 from the host, mem0_network stays a plain bridge — which also
    leaves mem0 general egress, so the pin is detected rather than prevented. See
    `test_the_pin_survives_without_the_internal_network` for what carries it instead.
    """
    networks = compose(override_yaml("gateway_models"))["networks"]
    assert not any((spec or {}).get("internal") for spec in networks.values())


def test_the_pin_survives_without_the_internal_network(suite: Suite) -> None:
    """With egress open, the model ids in .env and /configure are the whole pin.

    Nothing stops a bypass at the network layer any more, so both config surfaces must
    name gateway-routed models — `verify` catches a bypass after the fact by seeing no
    gateway traffic for the phase.
    """
    rendered = env_file(suite, key=None, password="pw", gateway_url=GATEWAY_URL)
    payload = configure_payload(suite, gateway_url=GATEWAY_URL)
    assert f"OPENAI_BASE_URL={GATEWAY_URL}" in rendered
    assert payload["llm"]["config"]["openai_base_url"] == GATEWAY_URL
    assert payload["embedder"]["config"]["openai_base_url"] == GATEWAY_URL


def test_the_override_drops_the_startup_pip_install() -> None:
    """Upstream's dev command refetches the SDK from PyPI at *every* container start.

    An internal network has no DNS, so pip retries and the container exits 1 before
    uvicorn is reached. This is what made `internal: true` and the stock command
    mutually exclusive.
    """
    service = compose(override_yaml())["services"]["mem0"]
    assert "pip install" not in service["command"]
    assert "rm -rf /app/packages" not in service["command"]


def test_the_override_still_migrates_and_serves() -> None:
    """Dropping the pip step must not drop the two commands that actually start mem0."""
    command = compose(override_yaml())["services"]["mem0"]["command"]
    assert "alembic upgrade head" in command
    assert "uvicorn main:app" in command


def test_the_override_remounts_the_sdk_the_bind_mount_hides() -> None:
    """`.:/app` covers the image's `/app/packages`, which is where `pip install -e` points.

    Without the source back at that path `import mem0` fails, which is why upstream
    reinstalls from PyPI. Remounting it is what makes the offline start work instead.
    """
    volumes = compose(override_yaml())["services"]["mem0"]["volumes"]
    assert any(v.endswith("/app/packages/mem0:ro") for v in volumes), volumes


def test_the_override_does_not_watch_the_bind_mount() -> None:
    """`--reload` restarts uvicorn on any file event, mid-run, for no benchmark benefit."""
    assert "--reload" not in compose(override_yaml())["services"]["mem0"]["command"]


def test_the_override_adds_no_service_of_its_own() -> None:
    """mem0's own published port is the ingress; nothing sits in front of it."""
    assert set(compose(override_yaml())["services"]) == {"mem0"}


def test_the_override_does_not_touch_the_stock_port_mappings() -> None:
    """Compose merges the override, so an absent `ports` key keeps what the base publishes.

    Restating a mapping the base file already declares would be a second place to change
    it. Both routes the host needs — 8888 for the API and 3000 for the dashboard — are
    upstream's, and they publish because nothing here resets them.
    """
    assert "ports" not in compose(override_yaml())["services"]["mem0"]
    assert "mem0-dashboard" not in compose(override_yaml())["services"]


@pytest.mark.skipif(not BASE_COMPOSE.is_file(), reason="mem0 checkout absent; --clone fetches it")
def test_the_stock_compose_publishes_both_ports_the_host_needs() -> None:
    """What the test above relies on upstream for, read from the checkout when present.

    Skipped rather than vendored: `systems/` is git-ignored (upstream's code under
    upstream's licence), so this cannot run on a fresh clone. It is the one check that
    would notice upstream dropping a mapping the override now depends on.
    """
    base = yaml.safe_load(BASE_COMPOSE.read_text())["services"]
    assert base["mem0"]["ports"] == ["8888:8000"]
    assert base["mem0-dashboard"]["ports"] == ["3000:3000"]


def test_reset_takes_the_volumes_down_with_the_containers() -> None:
    """Stopping is not resetting: without `-v` the pgvector volume survives untouched."""
    assert reset_argv() == ["docker", "compose", "down", "-v"]


def test_reset_removes_the_history_db_that_the_volume_drop_cannot_reach(tmp_path: Path) -> None:
    """`history/` is a host bind mount, so `down -v` leaves mem0's audit log behind."""
    assert reset_paths(tmp_path) == [tmp_path / "history" / "history.db"]


def test_reset_keeps_the_env_file_so_the_new_volume_initialises(tmp_path: Path) -> None:
    """The password in `.env` is what `init-db.sh` uses when the volume is recreated.

    Deleting `.env` here would rotate the password against a database about to be
    initialised with it, which is the one combination that cannot authenticate.
    """
    env = tmp_path / ".env"
    env.write_text("POSTGRES_PASSWORD=keepme\n")
    assert env not in reset_paths(tmp_path)
    assert postgres_password(env, None) == "keepme"


@pytest.mark.parametrize(
    ("state", "expected"),
    [("running", []), ("exited", ["mem0"]), ("restarting", [])],
)
def test_dead_services_names_only_the_ones_that_gave_up(state: str, expected: list[str]) -> None:
    """A container that exited will not come back, so waiting out the timeout is pointless.

    The first run of this script sat 300s against a container that had died in the first
    ten seconds, and reported a Docker Desktop 502 rather than the pip failure in its log.
    """
    line = json.dumps({"Service": "mem0", "State": state})
    assert dead_services(line) == expected


def test_the_default_checkout_lives_with_the_other_benchmarked_systems() -> None:
    """One directory holds every system's checkout, named as the repo names them."""
    assert DEFAULT_SERVER.parts[0] == "systems"


def test_the_default_checkout_is_git_ignored() -> None:
    """A third-party checkout is thousands of files; committing it is never intended.

    `*.env` already covers the credential inside it, so the gap this closes is the
    checkout itself flooding `git status`.
    """
    done = subprocess.run(["git", "check-ignore", "-q", str(DEFAULT_SERVER)], cwd=REPO, check=False)
    assert done.returncode == 0, f"{DEFAULT_SERVER} is not covered by .gitignore"


def test_the_default_checkout_cannot_shadow_an_installed_sdk() -> None:
    """A repo-root `mem0/` resolves as a namespace package and hides the real SDK.

    The repo root is on `sys.path`, so `import mem0` would find the checkout instead of
    site-packages — breaking a future in-process mem0 adapter in a way that looks like a
    bad install. Nesting the checkout keeps it off `sys.path` entirely.
    """
    assert len(DEFAULT_SERVER.parts) > 1
    assert DEFAULT_SERVER.parts[0] != "mem0"


def test_the_summary_never_prints_the_key(suite: Suite) -> None:
    """Never log a credential (CLAUDE.md); the key reaches mem0 through the file only."""
    assert "vk-secret" not in summary(suite, key="vk-secret", server=Path("/srv/mem0"))


def test_the_summary_says_when_no_key_is_in_play(suite: Suite) -> None:
    """Keyless is the normal case now, and it changes what the row can claim about cost."""
    assert "none" in summary(suite, key=None, server=Path("/srv/mem0")).lower()
