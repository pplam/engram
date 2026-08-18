"""Configure and start the self-hosted mem0 server against the gateway (steps 3-5).

`registry/mem0.yaml` says model pinning is not expressible from the registry: mem0
reads its LLM and embedder config from its own `.env` and from `POST /configure`, both
outside this repo. This script is the translation, so the pin comes from `suites/<v>.yaml`
rather than from someone retyping four model ids into a `.env`.

Three things it does that are easy to get wrong by hand:

- **The provider prefix.** Bifrost reads the segment before the first `/` as the provider,
  so mem0 must ask for `proxy/deepseek/deepseek-v4-flash-0731`. The bare id resolves to
  provider `deepseek` and fails (`gateway.wire_name`).
- **The embedding width.** pgvector writes `vector(N)` at `CREATE TABLE` and never widens
  it. mem0 defaults to 1536, suite v3 pins 4096, and the mismatch surfaces as a failed
  insert on every chunk rather than as a config error.
- **The gateway's network.** mem0 runs in its own compose project, so `OPENAI_BASE_URL`
  has no host to resolve until mem0 is attached to the gateway's `models` network. That
  network joins as external and must already exist — start the gateway first.

The pin here is *detected, not enforced.* An internal network would make it enforcement,
but Docker cannot publish a port for a container whose every network is internal, and the
harness dials mem0 on the host. So mem0 keeps ordinary egress, and what holds the pin is
the model ids written below plus `verify`: a phase with no gateway traffic went somewhere
else (ARCH §3.3).

A gateway running without governance needs no credential, so no key is provisioned by
default:

    uv run python -m scripts.setup_mem0 --clone

That is a weaker pin than a virtual key, and the difference is worth stating because it
lands in a published column. `allowed_models` no longer bounds what mem0 may ask for —
what remains is the model ids written below and the provider key's own `models` list in
`gateway/bifrost.json`.
And because `orchestrator/metrics.py` buckets usage by the `virtual_key_id` label, keyless
traffic carries no label and is invisible to the per-phase cost diff.

`--reset` starts from an empty store, dropping the Postgres volume and the history db that
the volume drop cannot reach:

    uv run python -m scripts.setup_mem0 --reset

Not the default, and not per run. Isolation is already a string — `user_id` embeds `run_id`
(ARCH §4.6) — so correctness never depends on a wipe, and a reset is global where a run is
namespaced: it would discard a concurrent run's memories mid-flight. What it is good for is
measurement hygiene before a publishable comparison set, so every system searches a
comparably-sized store, since §7.4 latency grows with the table.

Set `MEM0_GATEWAY_KEY` to restore both — it is written only into mem0's `.env`, never
echoed:

    curl -sS localhost:8080/api/governance/virtual-keys -H 'content-type: application/json' \\
      -d '{"name": "mem0:standing", "provider_configs": [...]}'

    MEM0_GATEWAY_KEY=<value> uv run python -m scripts.setup_mem0
"""

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from orchestrator.gateway import PinnedModel, wire_name
from orchestrator.suite import CURRENT_SUITE, Suite, load_suite

REPO_URL = "https://github.com/mem0ai/mem0.git"

# Where a benchmarked system's own checkout goes: `systems/<name>/`, git-ignored, one
# directory per system. Fetched rather than vendored, for the reason `datasets/*/data/`
# is — it is upstream's code under upstream's licence, verifiable from its own remote.
#
# Nested rather than a repo-root `mem0/` because the repo root is on `sys.path`: a
# top-level `mem0/` resolves as a namespace package and shadows the installed SDK, which
# would break a future in-process adapter in a way that reads like a bad install.
SYSTEMS = Path("systems")
DEFAULT_SERVER = SYSTEMS / "mem0" / "server"

# `gateway/docker-compose.yml` under compose project `gateway`, which names its internal
# network `models`. Overridable because a user may run the gateway under another project.
GATEWAY_NETWORK = "gateway_models"

# Reachable from inside `gateway_models`. The `/v1` suffix belongs here and not in
# `bifrost.json`: mem0 hands this to the OpenAI SDK, which appends only the path, whereas
# Bifrost adds the version prefix itself.
GATEWAY_URL = "http://gateway:8080/v1"

# Where mem0's own compose publishes the server: 8888 on the host, 8000 in the container.
MEM0_URL = "http://127.0.0.1:8888"

# What goes in OPENAI_API_KEY when the gateway wants no credential. Not a secret and not
# a fallback: mem0 constructs an `openai.OpenAI` client, which raises `OpenAIError` when
# no key is present anywhere, so the field has to hold *something* to get past __init__.
# Named to read as deliberate in a config dump rather than like a leaked placeholder.
NO_KEY_REQUIRED = "gateway-requires-no-key"

_PASSWORD_LINE = re.compile(r"^POSTGRES_PASSWORD=(.+)$", re.MULTILINE)


class SetupError(Exception):
    """The server could not be configured, or the suite does not pin enough to try."""


def _model(pinned: Any) -> str:
    """Return how mem0 must name a suite-pinned model in a request to the gateway."""
    return wire_name(PinnedModel(id=pinned.id, provider=pinned.provider))


def collection_name(suite: Suite) -> str:
    """Return the pgvector collection for `suite`, scoped so a width change gets a table.

    An existing table keeps its `vector(N)` width forever, so a suite that repins the
    embedder must not land in the previous suite's collection.
    """
    return f"engram_{suite.suite}"


def _dimensions(suite: Suite) -> int:
    if suite.models.embedding.dimensions is None:
        raise SetupError(
            f"suite {suite.suite} pins no embedding dimensions, and pgvector needs the "
            "width at CREATE TABLE — guessing would build a column the embedder cannot "
            "write into"
        )
    return suite.models.embedding.dimensions


def env_file(suite: Suite, key: str | None, password: str, gateway_url: str) -> str:
    """Return mem0's `.env`: Postgres, auth off, and every model call via the gateway."""
    return "\n".join(
        (
            f"# Generated by scripts/setup_mem0.py from suite {suite.suite}. Do not edit by",
            "# hand: the model ids and the gateway route are the suite's pin.",
            "POSTGRES_HOST=postgres",
            "POSTGRES_PORT=5432",
            "POSTGRES_DB=postgres",
            "POSTGRES_USER=postgres",
            f"POSTGRES_PASSWORD={password}",
            f"POSTGRES_COLLECTION_NAME={collection_name(suite)}",
            "",
            "# `POST /configure` is require_admin, and auth_type `disabled` is what admits",
            "# it without provisioning a user. Local container, no route in but ours.",
            "AUTH_DISABLED=true",
            "JWT_SECRET=unused-because-auth-is-disabled",
            "",
            "# mem0 phones home by default. The container has egress, so this would",
            "# succeed — which is the reason to turn it off rather than leave it to fail.",
            "MEM0_TELEMETRY=false",
            "",
            "# The gateway is the only route to a model (D5). Both halves of mem0 — the",
            "# fact-extraction LLM and the embedder — read OPENAI_BASE_URL.",
            f"OPENAI_BASE_URL={gateway_url}",
            f"OPENAI_API_KEY={key or NO_KEY_REQUIRED}",
            f"MEM0_DEFAULT_LLM_MODEL={_model(suite.models.chat)}",
            f"MEM0_DEFAULT_EMBEDDER_MODEL={_model(suite.models.embedding)}",
            "",
        )
    )


def override_yaml(gateway_network: str = GATEWAY_NETWORK) -> str:
    """Return a compose override attaching mem0 to the gateway and fixing its startup.

    An override rather than an edit to mem0's own compose file, so `git pull` in their
    repo stays clean and what we changed is legible in one place.
    """
    return "\n".join(
        (
            "# Generated by scripts/setup_mem0.py. Applied automatically by",
            "# `docker compose up` as docker-compose.override.yaml.",
            "#",
            "# Joining the gateway's network is what gives `OPENAI_BASE_URL=http://gateway:...`",
            "# a host to resolve. The gateway is a separate compose project, so its network",
            "# joins as external and must already exist — start the gateway first.",
            "#",
            "# mem0 keeps its own bridge network and the egress that comes with it, because a",
            "# container whose every network is `internal: true` cannot publish a port: the",
            "# host gets a Docker Desktop 502 and `docker port` lists nothing. The harness",
            "# dials 127.0.0.1:8888, so the port has to publish. That makes the model pin",
            "# detected rather than prevented: the model ids in `.env` and `POST /configure`",
            "# route every call through the gateway, and `verify` catches a bypass after the",
            "# fact — a phase with no gateway traffic went somewhere else (ARCH §3.3).",
            "#",
            "# `command` and the extra volume are unrelated to the network: upstream's dev",
            "# command runs `rm -rf /app/packages && pip install mem0ai` at *every* start,",
            "# because its own `.:/app` bind mount covers the image's `/app/packages` — the",
            "# path `pip install -e` recorded at build time — so `import mem0` fails without a",
            "# refetch. Remounting the SDK source at that exact path fixes the import, which",
            "# pins the SDK to the cloned commit instead of whatever PyPI serves at container",
            "# start — what a benchmark wants anyway, and one less network dependency in the",
            "# startup path. `--reload` goes too: it watches the bind mount and would restart",
            "# uvicorn mid-run.",
            "#",
            "# No `ports` here. Upstream already publishes 8888 for the API and 3000 for the",
            "# dashboard; restating either would be a second place to change it.",
            "services:",
            "  mem0:",
            f"    networks: [mem0_network, {gateway_network}]",
            "    volumes:",
            "      - ../mem0:/app/packages/mem0:ro",
            '    command: sh -c "alembic upgrade head && uvicorn main:app --host 0.0.0.0'
            ' --port 8000"',
            "networks:",
            f"  {gateway_network}:",
            "    external: true",
            "",
        )
    )


def configure_payload(suite: Suite, gateway_url: str = GATEWAY_URL) -> dict[str, Any]:
    """Return the `POST /configure` body pinning mem0's models, width, temperature, budget.

    The `.env` alone is not enough. It carries no dimension — mem0 would build a
    `vector(1536)` column — and no temperature, so `DEFAULT_CONFIG`'s hardcoded 0.2 would
    stand where the suite pins 0. It carries no completion budget either, and mem0's own
    default of 2000 is below what the pinned reasoning model spends on reasoning before it
    emits any content: the extraction returns empty, mem0 reads that as "no facts", and
    the chunk is silently not stored behind an HTTP 200.
    """
    dims = _dimensions(suite)
    llm_config: dict[str, Any] = {
        "model": _model(suite.models.chat),
        "temperature": suite.models.chat.temperature,
        "openai_base_url": gateway_url,
    }
    # Omitted rather than sent as null when the suite pins none: v1..v3 predate the pin,
    # and a null would replace mem0's own default with nothing at all.
    if suite.models.chat.max_tokens is not None:
        llm_config["max_tokens"] = suite.models.chat.max_tokens

    return {
        "version": "v1.1",
        "llm": {"provider": "openai", "config": llm_config},
        "embedder": {
            "provider": "openai",
            "config": {
                "model": _model(suite.models.embedding),
                "embedding_dims": dims,
                "openai_base_url": gateway_url,
            },
        },
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "host": "postgres",
                "port": 5432,
                "dbname": "postgres",
                "user": "postgres",
                "collection_name": collection_name(suite),
                "embedding_model_dims": dims,
                # pgvector caps both ANN index types at 2000 dimensions. mem0 guards
                # its diskann branch with that check and its hnsw branch with nothing,
                # so at 4096 `CREATE INDEX` raises ProgramLimitExceeded and every
                # `add` answers 502. Exact search is slower and is the only option at
                # this width; latency numbers are comparable across systems either
                # way, since no system gets an index it cannot build.
                "hnsw": False,
                "diskann": False,
            },
        },
    }


def postgres_password(env_path: Path, from_env: str | None) -> str:
    """Return the password to use, preferring an explicit one, then the existing `.env`.

    Reused rather than regenerated because the Postgres volume outlives the file: a fresh
    password on a second run authenticates against a database that still holds the first.
    """
    if from_env:
        return from_env
    if env_path.is_file():
        found = _PASSWORD_LINE.search(env_path.read_text())
        if found and found.group(1).strip():
            return found.group(1).strip()
    return secrets.token_urlsafe(24)


def summary(suite: Suite, key: str | None, server: Path) -> str:
    """Return what was configured. Never includes the key — only whether one is set."""
    credential = (
        f"virtual key from MEM0_GATEWAY_KEY ({len(key)} chars, not shown)"
        if key
        else "none — gateway requires no key, so this run's mem0 traffic carries no "
        "virtual_key_id and is outside the per-phase cost diff"
    )
    return "\n".join(
        (
            f"suite       {suite.suite}",
            f"server      {server}",
            f"llm         {_model(suite.models.chat)} @ temperature "
            f"{suite.models.chat.temperature}",
            f"embedder    {_model(suite.models.embedding)} @ {_dimensions(suite)} dims",
            f"collection  {collection_name(suite)}",
            f"credential  {credential}",
        )
    )


def _run(argv: list[str], cwd: Path | None = None) -> None:
    """Run `argv`, raising `SetupError` with its name on a non-zero exit."""
    try:
        subprocess.run(argv, cwd=cwd, check=True)
    except FileNotFoundError as err:
        raise SetupError(f"{argv[0]} is not on PATH") from err
    except subprocess.CalledProcessError as err:
        raise SetupError(f"`{' '.join(argv)}` failed with exit code {err.returncode}") from err


def _clone(server: Path) -> None:
    if (server / "docker-compose.yaml").is_file():
        return
    if server.exists() and any(server.iterdir()):
        raise SetupError(f"{server} exists and is not a mem0 server checkout")
    server.parent.parent.mkdir(parents=True, exist_ok=True)
    # `server/` is one directory of a large repo; depth 1 is enough to run it.
    _run(["git", "clone", "--depth", "1", REPO_URL, str(server.parent)])
    if not (server / "docker-compose.yaml").is_file():
        raise SetupError(f"cloned mem0 but found no compose file at {server}")


def reset_argv() -> list[str]:
    """Return the compose command that discards mem0's stored state.

    `-v` is the whole point: `down` alone stops the containers and leaves the pgvector
    volume — and so every ingested memory — exactly where it was.
    """
    return ["docker", "compose", "down", "-v"]


def reset_paths(server: Path) -> list[Path]:
    """Return the host paths a volume drop cannot reach, so a reset has to delete them.

    `history/` is a bind mount rather than a named volume, so `down -v` never touches
    mem0's audit log. `.env` is deliberately absent: the volume is recreated by
    `init-db.sh` using the password that file holds, so rotating it here would be the one
    combination that cannot authenticate (see `postgres_password`).
    """
    return [server / "history" / "history.db"]


def dead_services(ps_json: str) -> list[str]:
    """Return the services in `docker compose ps --format json` output that have exited.

    `restarting` is not dead: it is on its way back up. `exited` will not recover on its
    own, so waiting out the timeout only delays the log that says why.
    """
    names: list[str] = []
    for line in ps_json.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row: Any = json.loads(stripped)
        except ValueError:
            continue
        if isinstance(row, dict) and str(row.get("State", "")).lower() == "exited":
            names.append(str(row.get("Service", "")))
    return names


def _exited(server: Path) -> list[str]:
    """Return exited services, or nothing if compose cannot be asked."""
    try:
        done = subprocess.run(
            ["docker", "compose", "ps", "--all", "--format", "json"],
            cwd=server,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    return dead_services(done.stdout)


def _await_server(url: str, timeout_s: float, server: Path | None = None) -> None:
    """Poll until the server answers, so `/configure` is not sent into a boot sequence."""
    deadline = time.monotonic() + timeout_s
    last = "no response"
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{url}/docs", timeout=5)
        except httpx.HTTPError as err:
            last = str(err)
        else:
            if response.status_code // 100 in (2, 3, 4):
                return
            last = f"HTTP {response.status_code}"

        # A dead container will not answer later. Reported with its own log, because the
        # host-side symptom is misleading: Docker Desktop's forwarder answers 502 for a
        # container that is already gone, which says nothing about why it left.
        if server is not None and (gone := _exited(server)):
            raise SetupError(
                f"{', '.join(gone)} exited before serving; last from {url}: {last}\n"
                f"  docker compose -f {server / 'docker-compose.yaml'} logs mem0"
            )
        time.sleep(2)
    raise SetupError(f"mem0 did not come up at {url} within {timeout_s:.0f}s ({last})")


def _configure(url: str, payload: dict[str, Any]) -> None:
    try:
        response = httpx.post(f"{url}/configure", json=payload, timeout=60)
    except httpx.HTTPError as err:
        raise SetupError(f"cannot reach mem0 at {url} to configure it: {err}") from err
    if response.status_code // 100 != 2:
        raise SetupError(
            f"mem0 refused the configuration with HTTP {response.status_code}: {response.text}"
        )


def main(argv: list[str] | None = None) -> int:
    """Configure, start, and pin a mem0 server for one suite; return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--suite", default=CURRENT_SUITE)
    parser.add_argument("--gateway-network", default=GATEWAY_NETWORK)
    parser.add_argument("--gateway-url", default=GATEWAY_URL)
    parser.add_argument("--mem0-url", default=MEM0_URL)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--clone", action="store_true", help=f"clone {REPO_URL} if absent")
    parser.add_argument(
        "--write-only",
        action="store_true",
        help="write .env and the compose override, then stop without starting anything",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="discard every stored memory first: compose down -v, plus the history db",
    )
    args = parser.parse_args(argv)

    # Optional: a gateway without governance issues no keys, and an absent one is not an
    # error any more than it is for the harness itself (`orchestrator/cli.py:build_chat`).
    # Set it and both the model pin and the cost attribution come back.
    key = os.environ.get("MEM0_GATEWAY_KEY") or None

    try:
        suite = load_suite(args.suite, Path())
        if args.clone:
            _clone(args.server)
        if not (args.server / "docker-compose.yaml").is_file():
            raise SetupError(f"no mem0 compose file at {args.server} (pass --clone to fetch it)")

        env_path = args.server / ".env"
        password = postgres_password(env_path, os.environ.get("POSTGRES_PASSWORD"))
        payload = configure_payload(suite, args.gateway_url)

        # Before `.env` is rewritten, so the password read above is the one the recreated
        # volume initialises with. Nothing is namespaced here: a reset discards every run's
        # memories at once, which is why it is a flag and not the default (ARCH §4.6 —
        # isolation is `run_id` in `user_id`, and correctness does not depend on a wipe).
        if args.reset:
            _run(reset_argv(), cwd=args.server)
            for path in reset_paths(args.server):
                path.unlink(missing_ok=True)
            print(f"reset {args.server}: volumes dropped, history removed")

        # Written before anything starts, and 0600 either way: the Postgres password is
        # in it even when the gateway needs no key.
        env_path.write_text(env_file(suite, key, password, args.gateway_url))
        env_path.chmod(0o600)
        (args.server / "docker-compose.override.yaml").write_text(
            override_yaml(args.gateway_network)
        )
        print(summary(suite, key, args.server))

        if args.write_only:
            print(f"\nwrote {env_path} and the compose override; nothing started")
            if args.reset:
                print("mem0 is stopped — rerun without --write-only to bring it back up")
            return 0

        # Explicit, so a slow first pull reads as a pull rather than as `up` hanging.
        _run(["docker", "compose", "pull"], cwd=args.server)
        _run(["docker", "compose", "up", "-d"], cwd=args.server)
        _await_server(args.mem0_url, args.timeout, args.server)

        # After boot: `/configure` replaces the running config, and the dimension must be
        # in place before the first `add` creates the table.
        _configure(args.mem0_url, payload)
        print(f"\nconfigured {args.mem0_url} with:\n{json.dumps(payload, indent=2)}")
    except SetupError as err:
        print(f"{err}", file=sys.stderr)
        return 1

    print(
        "\nnext: uv run bench doctor mem0"
        f"\n      uv run bench run --system mem0 --dataset locomo-refined "
        f"--suite {suite.suite}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
