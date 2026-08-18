"""The `bench` CLI. Plain-text tables, stdlib formatting only."""

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import httpx

from contract.adapter import MemoryAdapter
from contract.conformance.checks import CheckResult, run_all
from orchestrator.artifacts import ArtifactError, read_json
from orchestrator.compose import (
    DEFAULT_COMPOSE,
    DEFAULT_CONFIG,
    DEFAULT_URL,
    ComposeError,
    credential_vars,
    is_ready,
    stop_argv,
    unset_credentials,
    up_argv,
    wait_ready,
)
from orchestrator.datasets import DEFAULT_DATASETS, DatasetError
from orchestrator.gateway import Gateway
from orchestrator.ids import IdError, new_run_id, resolve_prefix
from orchestrator.models import ChatClient, ModelError, OpenAiCompatibleChat
from orchestrator.ps import PsRow, collect_rows, format_stage, render_table
from orchestrator.registry import DEFAULT_REGISTRY, RegistryError, load_adapter, load_system
from orchestrator.report import Report
from orchestrator.run import (
    IN_PROCESS,
    ResumedConfig,
    RunError,
    RunRequest,
    check_resumable,
    execute_run,
    resumed_config,
)
from orchestrator.runlog import RUN_LOG
from orchestrator.runstate import read_state, resolve_status
from orchestrator.scheduler import Job, default_parallel, expand_jobs, run_jobs
from orchestrator.stages.score import run_score
from orchestrator.suite import CURRENT_SUITE, SuiteError
from orchestrator.systems import collect_systems
from orchestrator.systems import render_table as render_systems
from orchestrator.verify import Verification, run_verify
from orchestrator.worker import (
    ChildResult,
    child_argv,
    describe_failure,
    spawn,
    spawn_detached,
)
from reference.baselines import BASELINE_ADAPTERS, build_baseline

DEFAULT_ARTIFACTS = Path("artifacts")

# Where `bench gateway start` publishes the gateway. Model calls go here and nowhere else: the
# gateway is the only route to a model (D5), which makes it a constant rather than a flag.
# A run works as soon as the gateway is up, with nothing to configure.
#
# No `/v1` suffix: the chat client appends `/v1/chat/completions` itself, and a prefix here
# would send calls to `/v1/v1/chat/completions`, which the gateway answers with 405.
DEFAULT_MODELS_URL = DEFAULT_URL

# The one variable a harness model call reads its credential from. A fixed name rather
# than a flag: there is only ever one endpoint, so there is only ever one key to name.
# Unset is fine — see `build_chat`.
MODELS_KEY_ENV = "ENGRAM_GATEWAY_KEY"

# Home the cursor and erase the screen. `--watch` is a clear-and-redraw loop (§3.5.4).
CLEAR_SCREEN = "\033[H\033[2J"

# How much of `run.log` `bench logs` shows by default. An ingest of 2000 chunks writes a
# line per chunk, so dumping the whole file by default would bury the interesting end.
DEFAULT_TAIL = 40

# How long `--follow` sleeps when the log has nothing new.
FOLLOW_INTERVAL_S = 0.25


def _render(results: Sequence[CheckResult]) -> str:
    width = max(len(r.name) for r in results)
    lines = [f"{'CHECK':<{width}}  STATUS  DETAIL"]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"{r.name:<{width}}  {status:<6}  {r.reason or ''}".rstrip())
    failed = sum(1 for r in results if not r.passed)
    lines.append("")
    lines.append(f"{len(results) - failed}/{len(results)} checks passed")
    return "\n".join(lines)


GATEWAY_HINT = (
    "warning: {system} is an in-process adapter and its config does not point at the "
    "gateway, so its model traffic cannot be metered or pinned. Its cost column will "
    "read `unenforced`. Pass the gateway through the adapter's `config` to fix it."
)


def _warn_if_unmetered(system: str, config: dict[str, object]) -> None:
    """Warn when an in-process adapter was not told to use the gateway (ARCH §3).

    A warning rather than a failure: we cannot prove where a library sends its traffic,
    only that it was never *told* to use the gateway. This catches the accident and not
    the deliberate case, which is exactly why those rows say `unenforced` rather than
    claiming enforcement.
    """
    haystack = " ".join(f"{k}={v}" for k, v in config.items()).lower()
    if config and ("gateway" in haystack or "base_url" in haystack or "8080" in haystack):
        return
    print(GATEWAY_HINT.format(system=system), file=sys.stderr)


def systems(registry: Path = DEFAULT_REGISTRY, as_json: bool = False) -> int:
    """List every name `bench run --system` accepts; return an exit code.

    Always 0, even when an entry is malformed: the listing reported it, and reporting it
    is the whole job. `bench doctor` is what fails on a system that cannot be built.
    """
    rows = collect_systems(registry)
    if as_json:
        print(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))
        return 0

    print(render_systems(rows))
    return 0


def doctor(system: str, registry: Path, run_id: str | None = None) -> int:
    """Run the conformance checks against one registered system; return an exit code."""
    probe_run_id = run_id or new_run_id()
    try:
        if system in BASELINE_ADAPTERS:
            adapter: MemoryAdapter = build_baseline(system)
        else:
            spec = load_system(system, registry)
            adapter = load_adapter(spec)
            _warn_if_unmetered(system, spec.config)
    except RegistryError as err:
        print(f"registry error: {err}", file=sys.stderr)
        return 2

    async def probe() -> list[CheckResult]:
        try:
            return await run_all(adapter, run_id=probe_run_id)
        finally:
            closer = getattr(adapter, "close", None)
            if closer is not None:
                await closer()

    results = asyncio.run(probe())
    print(_render(results))
    return 1 if any(not r.passed for r in results) else 0


def compose_launch(argv: list[str]) -> int:
    """Run `argv` to completion, inheriting stdio; return its exit code. Tests patch this.

    Output is inherited rather than captured: `docker compose` reports image pulls and
    container starts as it goes, and that is exactly what the user wants to watch.
    """
    return subprocess.run(argv, check=False).returncode


def gateway_transport() -> httpx.AsyncBaseTransport | None:
    """Return the transport for the readiness poll. Tests patch this; production is None."""
    return None


def _compose(argv: list[str]) -> int:
    """Run one `docker compose` argv, reporting its two user-facing failures."""
    try:
        code = compose_launch(argv)
    except FileNotFoundError:
        print("docker is not on PATH: install Docker to run the gateway", file=sys.stderr)
        return 2
    if code != 0:
        print(f"docker compose exited {code}", file=sys.stderr)
        return 1
    return 0


def gateway_start(
    compose_file: Path = DEFAULT_COMPOSE,
    config_file: Path = DEFAULT_CONFIG,
    url: str = DEFAULT_URL,
    wait: bool = True,
    timeout: float = 60.0,
    skip_env_check: bool = False,
) -> int:
    """Start the model plane and wait for it to answer; return an exit code."""
    if not compose_file.is_file():
        print(f"no compose file at {compose_file}", file=sys.stderr)
        return 2

    # Checked before starting anything: a gateway booted without its provider key comes
    # up healthy and 401s every model call, which only surfaces as a failed run later.
    #
    # A *partly* configured gateway is legitimate, though — the config carries one block
    # per provider and a dataset uses one of them, so a missing judge endpoint must not
    # block an OpenAI-only run. Hence a warning for some missing, a refusal for all.
    if not skip_env_check:
        try:
            required = credential_vars(config_file)
        except ComposeError as err:
            print(f"{err}", file=sys.stderr)
            return 2
        # Names only. The whole point of `env.NAME` in the config is that no value ever
        # needs to be read here.
        missing = unset_credentials(required, os.environ)
        if missing and len(missing) == len(required):
            print(
                f"no provider credentials set: {', '.join(missing)} — export at least "
                "one, or pass --skip-env-check if this gateway needs none",
                file=sys.stderr,
            )
            return 2
        if missing:
            print(
                f"warning: unset provider credentials: {', '.join(missing)}. Any model "
                "served by those providers will fail; the rest of the gateway works.",
                file=sys.stderr,
            )

    code = _compose(up_argv(compose_file))
    if code != 0:
        return code

    if wait:
        try:
            asyncio.run(_await_gateway(url, timeout))
        except ComposeError as err:
            print(f"{err}", file=sys.stderr)
            return 1

    print(f"gateway up at {url} — `bench run` uses it with no further configuration")
    return 0


def gateway_stop(compose_file: Path = DEFAULT_COMPOSE) -> int:
    """Stop the model plane; return an exit code.

    No credential check and no readiness poll: both exist to catch a start that would look
    healthy and fail later, and neither says anything about stopping.
    """
    if not compose_file.is_file():
        print(f"no compose file at {compose_file}", file=sys.stderr)
        return 2

    code = _compose(stop_argv(compose_file))
    if code != 0:
        return code

    print("gateway stopped")
    return 0


def _probe_client(url: str, timeout_s: float) -> httpx.AsyncClient:
    """Return a client for the readiness probe, honouring the test transport seam.

    Per-request timeout is capped by the overall budget: a single hung connect must not
    outlast the deadline the user asked for.
    """
    transport = gateway_transport()
    return (
        httpx.AsyncClient(base_url=url, timeout=timeout_s, transport=transport)
        if transport is not None
        else httpx.AsyncClient(base_url=url, timeout=timeout_s)
    )


async def _await_gateway(url: str, timeout_s: float) -> None:
    async with _probe_client(url, timeout_s) as client:
        await wait_ready(client, timeout_s=timeout_s)


async def _probe_gateway(url: str, timeout_s: float) -> bool:
    async with _probe_client(url, timeout_s) as client:
        return await is_ready(client)


def gateway_status(url: str = DEFAULT_URL, timeout: float = 5.0) -> int:
    """Report whether the model plane is answering; return 0 when up, 1 when down.

    Reads no compose file and starts nothing, so it works against a gateway someone else
    started — a hosted one, or a container outside this repo's compose file. The exit code
    is the point: `bench gateway status && bench run ...` must not run against nothing.
    """
    if asyncio.run(_probe_gateway(url, timeout)):
        print(f"gateway up at {url}")
        return 0
    print(f"gateway down at {url} — start it with `bench gateway start`")
    return 1


def _render_row(report: Report) -> str:
    """Render one report row as aligned plain text."""
    row = report.row
    fields = [
        ("system", row.system),
        ("dataset", row.dataset),
        ("suite", row.suite),
        ("run_id", row.run_id),
        ("scorer", row.scorer),
        ("accuracy", f"{row.quality.accuracy:.3f} ({row.quality.correct}/{row.quality.questions})"),
        ("judge", row.quality.judge or "-"),
        ("recall@k", _number(row.retrieval.recall_at_k)),
        ("recall track", f"{row.retrieval.track} / {row.retrieval.evidence_unit}"),
        (
            "harness tokens",
            str(row.cost.harness_prompt_tokens + row.cost.harness_completion_tokens),
        ),
        ("system cost", row.cost.system_attribution),
        ("in_process", str(row.in_process)),
    ]
    width = max(len(name) for name, _ in fields)
    lines = [f"{name:<{width}}  {value}" for name, value in fields]
    lines.append("")
    lines.extend(f"note: {note}" for note in report.notes)
    return "\n".join(lines)


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def model_transport() -> httpx.AsyncBaseTransport | None:
    """Return the transport for model calls. Tests patch this; production returns None."""
    return None


def gateway_client(url: str = DEFAULT_URL, timeout: float = 30.0) -> Gateway | None:
    """Return a client to the running gateway, for keys and metering. Tests patch this.

    Starts nothing. `bench gateway start` runs the container and it stays up across runs; this
    only opens a connection pool to it, so N runs share one gateway and are separated by
    their virtual keys rather than by processes.

    Returning None is not a supported production path — the caller degrades to unmetered
    when provisioning fails, which covers a gateway that is down or has governance off.
    The Optional is here so a test can disable metering through the same seam.
    """
    return Gateway(httpx.AsyncClient(base_url=url, timeout=timeout))


def build_chat(
    api_key_env: str | None = MODELS_KEY_ENV,
    timeout: float = 600.0,
    gateway_url: str = DEFAULT_MODELS_URL,
) -> ChatClient:
    """Return the chat client for a run: always the gateway.

    Nothing to configure. The gateway is the only route to a model (D5), so pointing a run
    somewhere else is not a supported configuration — it would be an unmetered, unpinned
    run wearing the same report shape as a metered one. The credential is read from one
    fixed variable for the same reason: a run either goes through the gateway or it does
    not, and there is no second endpoint whose key could need naming.

    An unset credential is not an error. The gateway holds the provider keys; the harness
    presents a virtual key when it has one, and a gateway with governance disabled needs
    none at all. A missing one surfaces as the gateway's own 401, which says more than a
    guess made here.
    """
    key = os.environ.get(api_key_env) if api_key_env else None

    transport = model_transport()
    client = (
        httpx.AsyncClient(base_url=gateway_url, timeout=timeout, transport=transport)
        if transport is not None
        else httpx.AsyncClient(base_url=gateway_url, timeout=timeout)
    )
    return OpenAiCompatibleChat(client, api_key=key)


def _adapter_for(system: str, registry: Path) -> MemoryAdapter:
    """Return the adapter for one system: in-repo baselines directly, others by registry.

    Both paths go through the identical two methods from here on — a baseline that took
    a shortcut would not be measuring the same thing.
    """
    if system in BASELINE_ADAPTERS:
        return build_baseline(system)
    return load_adapter(load_system(system, registry))


async def _one_run(
    request: RunRequest,
    registry: Path,
    chat: ChatClient,
    judge_chat: ChatClient,
) -> Report:
    adapter = None if request.is_in_process else _adapter_for(request.system, registry)
    gateway = gateway_client()
    if gateway is None:
        return await execute_run(request, adapter, chat=chat, judge_chat=judge_chat)
    # Closed here rather than in `execute_run`, which takes the gateway as a parameter and
    # does not own it — a caller passing a shared client would have it closed underneath.
    async with gateway:
        return await execute_run(
            request, adapter, chat=chat, judge_chat=judge_chat, gateway=gateway
        )


def run(
    system: str | None,
    dataset: str | None,
    suite: str | None,
    artifacts: Path,
    datasets: Path,
    repo_root: Path,
    registry: Path,
    limit: int | None = None,
    parallel: int | None = None,
    allow_concurrent_same_system: bool = False,
    contended: bool = False,
    resume: str | None = None,
    name: str | None = None,
    run_id_override: list[str] | None = None,
    foreground: bool = False,
) -> int:
    """Launch one run, or a cross product of them, detaching unless `foreground`."""
    # No failure mode left to handle: the endpoint is a constant and a missing credential
    # is the gateway's 401 to report, not ours to pre-empt.
    chat = build_chat()
    judge_chat = build_chat()

    resumed: ResumedConfig | None = None
    if resume:
        # The run's own manifest says what it is, so a resume can supply the configuration
        # the user would otherwise retype. Resolved before the checks below so an omitted
        # system or dataset is filled in rather than read as "named nothing".
        try:
            run_id = resolve_prefix(resume, artifacts)
            resumed = resumed_config(artifacts / run_id)
        except (IdError, RunError) as err:
            print(f"{err}", file=sys.stderr)
            return 2
        system = system or resumed.system
        dataset = dataset or resumed.dataset
        # An explicit flag still wins, and `check_resumable` refuses it if it disagrees:
        # silently overriding what the user typed would be worse than the friction.
        suite = suite or resumed.suite
        limit = limit if limit is not None else resumed.limit
        name = name or resumed.name
    else:
        if system is None or dataset is None:
            print(
                "a system and --dataset are required, or --resume <id> to read them "
                "from an existing run's manifest",
                file=sys.stderr,
            )
            return 2
        # A fresh run pins the current suite; only a resume inherits an older one.
        suite = suite or CURRENT_SUITE

    systems = [s.strip() for s in system.split(",") if s.strip()]
    sets = [d.strip() for d in dataset.split(",") if d.strip()]
    if not systems or not sets:
        print(
            "--dataset and the system argument must each name at least one value",
            file=sys.stderr,
        )
        return 2

    # A run is only contended if the override is on *and* it actually shares its system
    # with a sibling. One run cannot contend with itself, and labelling it as though it
    # did would withhold a cost number that is perfectly sound.
    shared = allow_concurrent_same_system and len(sets) > 1

    if len(systems) > 1 or len(sets) > 1:
        if resume:
            # Resume targets exactly one run: its manifest names one system and one
            # dataset, and splicing a cross product into it is what --resume refuses.
            print("--resume takes a single system and dataset", file=sys.stderr)
            return 2
        try:
            jobs = expand_jobs(systems, sets, contended=shared, run_ids=run_id_override)
        except ValueError as err:
            print(f"{err}", file=sys.stderr)
            return 2
        # Ids up front, so a batch interrupted halfway — or detached and forgotten — is
        # still resumable by ids the user has seen. Printed by whichever process the user
        # is watching: the launcher when detaching, the supervisor when it is foreground.
        for job in jobs:
            print(f"{job.run_id}  {job.system}  {job.dataset}")

        if not foreground:
            return _detach_batch(
                jobs,
                suite=suite,
                artifacts=artifacts,
                datasets=datasets,
                repo_root=repo_root,
                registry=registry,
                limit=limit,
                parallel=parallel,
                allow_concurrent_same_system=allow_concurrent_same_system,
                name=name,
            )

        return _run_batch(
            jobs,
            suite=suite,
            artifacts=artifacts,
            datasets=datasets,
            repo_root=repo_root,
            registry=registry,
            limit=limit,
            parallel=parallel,
            allow_concurrent_same_system=allow_concurrent_same_system,
            name=name,
        )

    if resume:
        # `run_id` is already resolved above. Still checked, because an explicitly passed
        # system, dataset, or suite may contradict the manifest.
        try:
            check_resumable(
                artifacts / run_id,
                system=systems[0],
                dataset=sets[0],
                suite=suite,
                limit=limit,
            )
        except RunError as err:
            print(f"{err}", file=sys.stderr)
            return 2
    elif run_id_override:
        # A batch parent dictated the id so it can label this child's output. One value
        # here: the flag is repeatable only so a detached batch can pass its whole id list
        # to a supervisor, and that path never reaches this single-run branch.
        if len(run_id_override) > 1:
            print("--run-id takes one value for a single run", file=sys.stderr)
            return 2
        run_id = run_id_override[0]
    else:
        run_id = new_run_id()
        # Printed before the work starts, so an interrupted run is still resumable by
        # an id the user has already seen.
        print(run_id)

    if not foreground:
        # A run takes hours, so holding the terminal for it makes a closed lid or a stray
        # Ctrl-C the end of the run. The child records its own progress; this process is
        # done as soon as it exists. Everything above already ran, so a usage error is
        # reported here rather than dying unseen in a child.
        job = Job(run_id=run_id, system=systems[0], dataset=sets[0], contended=contended)
        argv = child_argv(
            job,
            suite=suite,
            artifacts=artifacts,
            datasets=datasets,
            repo_root=repo_root,
            registry=registry,
            limit=limit,
            name=name,
        )
        if resume:
            argv += ["--resume", run_id]
        try:
            spawn_detached(argv, cwd=repo_root)
        except OSError as err:
            print(f"cannot start {run_id}: {err}", file=sys.stderr)
            return 1
        if resume:
            print(run_id)
        print(
            f"detached. `bench ps` for progress, `bench logs {run_id}` for the log.",
            file=sys.stderr,
        )
        return 0

    request = RunRequest(
        run_id=run_id,
        system=systems[0],
        dataset=sets[0],
        suite=suite,
        artifacts_root=artifacts,
        datasets_root=datasets,
        repo_root=repo_root,
        limit=limit,
        contended=contended,
        name=name,
    )
    try:
        report = asyncio.run(_one_run(request, registry, chat, judge_chat))
    except RegistryError as err:
        print(f"registry error: {err}", file=sys.stderr)
        return 2
    except ModelError as err:
        # The gateway not being up is the likeliest failure now that it is the default,
        # and there is no flag to point elsewhere — so the message names the fix. The run
        # stays resumable: artifacts are append-only and completed stages are skipped.
        print(
            f"{run_id} failed: {err}\n"
            f"Model calls go to {DEFAULT_MODELS_URL}. Start the model plane with "
            f"`bench gateway start`, then resume with `bench run --resume {run_id}`.",
            file=sys.stderr,
        )
        return 1
    except (RunError, ArtifactError, DatasetError, SuiteError) as err:
        print(f"{run_id} failed: {err}", file=sys.stderr)
        return 1

    print(_render_row(report))
    return 0


def _detach_batch(
    jobs: list[Job],
    suite: str,
    artifacts: Path,
    datasets: Path,
    repo_root: Path,
    registry: Path,
    limit: int | None,
    parallel: int | None,
    allow_concurrent_same_system: bool,
    name: str | None = None,
) -> int:
    """Start the whole cross product in a detached supervisor; return an exit code.

    One supervisor rather than one detached child per job. Detaching the jobs themselves
    would drop `run_jobs`' per-system lock — the parent's `async with` would fall through
    the instant a child was launched rather than when it finished — so two runs against one
    system could overlap and contaminate latency and rank stability (§7.4, §7.5). The
    supervisor holds those locks, and only the *terminal* is released.

    The supervisor is this same command with `--foreground` and the ids already printed,
    so there is one batch code path rather than two. Its stdout goes nowhere, which is why
    the launcher prints the ids and why the aggregated report table is a `--foreground`
    feature: `bench ps`, `bench logs`, and each run's `report.json` are the detached view.
    """
    argv = [
        sys.executable,
        "-m",
        "orchestrator.cli",
        "run",
        ",".join(dict.fromkeys(job.system for job in jobs)),
        "--dataset",
        ",".join(dict.fromkeys(job.dataset for job in jobs)),
        "--suite",
        suite,
        "--foreground",
        "--artifacts",
        str(artifacts),
        "--datasets",
        str(datasets),
        "--repo-root",
        str(repo_root),
        "--registry",
        str(registry),
    ]
    # In cross-product order, which is the order `expand_jobs` reproduces from them.
    for job in jobs:
        argv += ["--run-id", job.run_id]
    if limit is not None:
        argv += ["--limit", str(limit)]
    if parallel is not None:
        argv += ["--parallel", str(parallel)]
    if allow_concurrent_same_system:
        argv.append("--allow-concurrent-same-system")
    if name is not None:
        argv += ["--name", name]

    try:
        spawn_detached(argv, cwd=repo_root)
    except OSError as err:
        print(f"cannot start the batch: {err}", file=sys.stderr)
        return 1

    print(
        f"detached {len(jobs)} runs. `bench ps` for progress, `bench logs <id>` for one run's log.",
        file=sys.stderr,
    )
    return 0


def _run_batch(
    jobs: list[Job],
    suite: str,
    artifacts: Path,
    datasets: Path,
    repo_root: Path,
    registry: Path,
    limit: int | None,
    parallel: int | None,
    allow_concurrent_same_system: bool,
    name: str | None = None,
) -> int:
    """Run a cross product, one subprocess per job, and print each child's row (§3.5.3).

    Blocking, so it is either what `--foreground` asked for or the body of a detached
    supervisor. The caller has already printed the ids.
    """
    outputs: dict[str, ChildResult] = {}

    async def work(job: Job) -> None:
        argv = child_argv(
            job,
            suite=suite,
            artifacts=artifacts,
            datasets=datasets,
            repo_root=repo_root,
            registry=registry,
            limit=limit,
            name=name,
        )
        result = await spawn(argv, cwd=repo_root)
        outputs[job.run_id] = result
        if not result.ok:
            raise RunError(describe_failure(result))

    results = asyncio.run(
        run_jobs(
            jobs,
            work,
            parallel=parallel if parallel is not None else default_parallel(jobs),
            allow_concurrent_same_system=allow_concurrent_same_system,
        )
    )
    failures = {job_id: str(err) for job_id, err in results.items() if err is not None}

    for job in jobs:
        result = outputs.get(job.run_id)
        if result is not None and result.ok:
            print(f"=== {job.run_id} ===")
            print(result.stdout.rstrip("\n"))
            print()

    for job_id, message in failures.items():
        print(f"{job_id} failed: {message}", file=sys.stderr)
        child = outputs.get(job_id)
        if child is not None and child.stderr.strip():
            print(child.stderr.rstrip("\n"), file=sys.stderr)

    if not failures:
        return 0
    # A misnamed system is a usage error, not a run that went wrong. Only report the usage
    # code when nothing ran at all, so a partly-successful batch still reports failure.
    succeeded = any(r.ok for r in outputs.values())
    every_failure_is_usage = all(
        outputs.get(job_id) is not None and outputs[job_id].returncode == 2 for job_id in failures
    )
    if not succeeded and every_failure_is_usage:
        return 2
    return 1


def rescore(run_id: str, artifacts: Path) -> int:
    """Recompute `report.json` from a finished run's artifacts; return an exit code."""
    run_dir = artifacts / run_id
    if not (run_dir / "manifest.json").is_file():
        print(f"no run {run_id!r} under {artifacts}", file=sys.stderr)
        return 2

    # Provenance is a property of the run, so it is read back from the manifest rather
    # than re-derived: a rescore must not be able to reclassify a published row.
    manifest = read_json(run_dir / "manifest.json")
    system = str(manifest["system"]["name"])

    try:
        report = run_score(run_dir, in_process=system in IN_PROCESS)
    except (ArtifactError, KeyError) as err:
        print(f"rescore failed: {err}", file=sys.stderr)
        return 1

    print(_render_row(report))
    return 0


def _render_verification(result: Verification) -> str:
    """Render `verify.json` as an aligned plain-text table."""
    width = max(len(f.name) for f in result.findings)
    lines = [f"{'CHECK':<{width}}  STATUS  SEVERITY         DETAIL"]
    for finding in result.findings:
        status = "PASS" if finding.passed else "FAIL"
        row = f"{finding.name:<{width}}  {status:<6}  {finding.severity:<15}  {finding.detail}"
        lines.append(row.rstrip())
    lines.append("")
    lines.append(f"run_valid        {result.run_valid}")
    lines.append(f"row_withheld     {result.row_withheld}")
    lines.append(f"cost attribution {result.cost_attribution}")
    return "\n".join(lines)


def verify(run_id: str, artifacts: Path, metered: bool = False, contended: bool = False) -> int:
    """Replay the verification checks over a finished run; return an exit code."""
    run_dir = artifacts / run_id
    if not (run_dir / "manifest.json").is_file():
        print(f"no run {run_id!r} under {artifacts}", file=sys.stderr)
        return 2

    try:
        result = run_verify(run_dir, usage={}, metered=metered, contended=contended)
    except (ArtifactError, KeyError) as err:
        print(f"verify failed: {err}", file=sys.stderr)
        return 1

    print(_render_verification(result))
    # A withheld row is not a clean run, so it must not exit 0.
    return 0 if result.run_valid and not result.row_withheld else 1


def ps(
    artifacts: Path,
    show_all: bool = False,
    as_json: bool = False,
    watch: bool = False,
    interval: float = 2.0,
) -> int:
    """List runs found under the artifacts root; return an exit code."""
    if watch and as_json:
        # A redraw loop emitting one JSON document per pass is not parseable output.
        print("--watch cannot be combined with --json", file=sys.stderr)
        return 2

    if watch:
        return _watch_ps(artifacts, show_all=show_all, interval=interval)

    rows = _ps_rows(artifacts, show_all=show_all)
    if as_json:
        print(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))
        return 0

    print(render_table(rows))
    return 0


def _ps_rows(artifacts: Path, show_all: bool) -> list[PsRow]:
    rows = collect_rows(artifacts)
    return rows if show_all else [row for row in rows if row.is_live]


def _watch_ps(artifacts: Path, show_all: bool, interval: float) -> int:
    """Clear and redraw the table until no run is live, or until interrupted.

    Plain ANSI and a sleep — a table this simple does not justify a curses or `rich`
    dependency, and the loop holds no state of its own: every pass is a fresh directory
    scan, so what it shows is exactly what `bench ps` would show once.
    """
    try:
        while True:
            rows = _ps_rows(artifacts, show_all=show_all)
            print(CLEAR_SCREEN + render_table(rows))
            if not any(row.is_live for row in collect_rows(artifacts)):
                return 0
            time.sleep(interval)
    except KeyboardInterrupt:
        # Ctrl-C on a viewer is how you leave it, not an error.
        return 0


def rm(run_id: str, artifacts: Path) -> int:
    """Delete one run's artifact directory; refuse while it is still running."""
    run_dir = artifacts / run_id
    if not run_dir.is_dir():
        print(f"no run {run_id!r} under {artifacts}", file=sys.stderr)
        return 2

    state = read_state(run_dir)
    if state is not None and resolve_status(state) in ("running", "queued"):
        # Removing a live run's directory would corrupt the artifacts it is appending to.
        print(f"run {run_id!r} is still running; stop it first", file=sys.stderr)
        return 1

    shutil.rmtree(run_dir)
    print(f"removed {run_id}")
    return 0


def logs(run_id: str, artifacts: Path, tail: int = DEFAULT_TAIL, follow: bool = False) -> int:
    """Print one run's status and the tail of its log; return an exit code.

    Now that runs detach, this is how a run is observed, so it prints the log itself and
    not only the four summary fields. `--follow` waits on the file the way `tail -f` does,
    which is the attached view a foreground run used to give.
    """
    run_dir = artifacts / run_id
    state = read_state(run_dir)
    if state is None:
        print(f"no readable state for run {run_id!r} under {artifacts}", file=sys.stderr)
        return 2

    print(f"run_id  {state.run_id}")
    print(f"system  {state.system}")
    print(f"status  {resolve_status(state)}")
    print(f"stage   {format_stage(state.stage, state.stage_index, state.stage_total)}")
    if state.error:
        print(f"error   {state.error}")

    path = run_dir / RUN_LOG
    if not path.is_file():
        # A run writes its state file before its first log record, so this is the normal
        # state of a run that has only just started — not a failure.
        print(f"(no log yet at {path})")
        return 0

    print()
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-tail:] if tail > 0 else lines:
        print(line)

    if follow:
        _follow(path, run_dir)
    return 0


def _follow(path: Path, run_dir: Path) -> None:
    """Print new log lines until the run stops or the user interrupts.

    Stops on its own when the run is no longer live, so `--follow` on a finished run
    returns instead of waiting for a writer that will never come back.
    """
    with path.open(encoding="utf-8", errors="replace") as handle:
        handle.seek(0, os.SEEK_END)
        try:
            while True:
                line = handle.readline()
                if line:
                    print(line.rstrip("\n"), flush=True)
                    continue
                state = read_state(run_dir)
                if state is None or resolve_status(state) not in ("running", "queued"):
                    return
                time.sleep(FOLLOW_INTERVAL_S)
        except KeyboardInterrupt:
            # Ctrl-C stops the watching, never the run. That distinction is the point of
            # detaching: `bench stop` is what ends a run.
            print()


def stop(run_id: str, artifacts: Path) -> int:
    """Signal a running run to stop; return an exit code."""
    run_dir = artifacts / run_id
    state = read_state(run_dir)
    if state is None:
        print(f"no readable state for run {run_id!r} under {artifacts}", file=sys.stderr)
        return 2
    if resolve_status(state) not in ("running", "queued"):
        print(f"run {run_id!r} is not running", file=sys.stderr)
        return 1

    try:
        # SIGINT rather than SIGKILL: a run drains gracefully and stays resumable by ID.
        os.kill(state.pid, signal.SIGINT)
    except (ProcessLookupError, PermissionError) as err:
        print(f"cannot signal pid {state.pid}: {err}", file=sys.stderr)
        return 1

    print(f"signalled {run_id} (pid {state.pid})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the `bench` argument parser."""
    parser = argparse.ArgumentParser(prog="bench", description="Engram memory benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    sys_cmd = sub.add_parser("systems", help="list every system `run --system` accepts")
    sys_cmd.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help=f"registry directory (default: {DEFAULT_REGISTRY})",
    )
    sys_cmd.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    doc = sub.add_parser("doctor", help="check a registered system against the contract")
    doc.add_argument("system", help="registry entry name")
    doc.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help=f"registry directory (default: {DEFAULT_REGISTRY})",
    )
    doc.add_argument(
        "--run-id",
        default=None,
        help="namespace for probe writes (default: a generated one, so probes never collide)",
    )

    gw = sub.add_parser("gateway", help="start or stop the model plane")
    gw_sub = gw.add_subparsers(dest="gateway_command", required=True)

    gw_start = gw_sub.add_parser("start", help="start the model plane and wait for it")
    gw_start.add_argument(
        "--compose",
        type=Path,
        default=DEFAULT_COMPOSE,
        help=f"compose file defining the gateway (default: {DEFAULT_COMPOSE})",
    )
    gw_start.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"gateway config, read for the credentials it references (default: {DEFAULT_CONFIG})",
    )
    gw_start.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"where the gateway is published (default: {DEFAULT_URL})",
    )
    gw_start.add_argument(
        "--no-wait", action="store_true", help="return as soon as compose does, without polling"
    )
    gw_start.add_argument(
        "--timeout", type=float, default=60.0, help="seconds to wait for readiness (default: 60)"
    )
    gw_start.add_argument(
        "--skip-env-check",
        action="store_true",
        help="start even when a referenced provider credential is unset",
    )

    gw_stop = gw_sub.add_parser("stop", help="stop the model plane, leaving the container in place")
    gw_stop.add_argument(
        "--compose",
        type=Path,
        default=DEFAULT_COMPOSE,
        help=f"compose file defining the gateway (default: {DEFAULT_COMPOSE})",
    )

    gw_status = gw_sub.add_parser("status", help="report whether the model plane is answering")
    gw_status.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"where the gateway is published (default: {DEFAULT_URL})",
    )
    gw_status.add_argument(
        "--timeout", type=float, default=5.0, help="seconds to wait for an answer (default: 5)"
    )

    run_cmd = sub.add_parser("run", help="run one system against one dataset")
    # Both are optional only so `--resume <id>` can read them from the run's own manifest;
    # `run` refuses a fresh run that names neither.
    run_cmd.add_argument(
        "system",
        nargs="?",
        default=None,
        help="registry entry name, or an in-process baseline; omit when using --resume",
    )
    run_cmd.add_argument("--dataset", default=None, help="dataset name; omit when using --resume")
    run_cmd.add_argument(
        # Defaulted to None rather than CURRENT_SUITE so a resume can tell "not passed"
        # from "passed explicitly" — otherwise resuming a run pinned to an older suite
        # would silently request the current one and be refused.
        "--suite",
        default=None,
        help=f"suite version (default: {CURRENT_SUITE}, or the manifest's under --resume)",
    )
    run_cmd.add_argument(
        "--resume",
        default=None,
        help=(
            "continue an existing run by id or unambiguous prefix; reads the system, "
            "dataset, suite, and limit from its manifest, and refuses any that is passed "
            "explicitly and disagrees"
        ),
    )
    run_cmd.add_argument(
        "--name",
        default=None,
        help="optional human label for `bench ps`; metadata, so it need not be unique",
    )
    run_cmd.add_argument(
        # Set by a batch parent so it can label the child's output, and by a detached
        # batch launcher to hand its supervisor the ids it already printed. Repeatable for
        # that second case: one per job, in cross-product order. A user never passes this
        # — ids are generated (§4.5) — so it is suppressed from the help text.
        "--run-id",
        action="append",
        default=None,
        help=argparse.SUPPRESS,
    )
    run_cmd.add_argument(
        "--foreground",
        action="store_true",
        help=(
            "stay attached and print the report row when the run finishes; the default "
            "detaches and returns the run id, so a run outlives its terminal"
        ),
    )
    run_cmd.add_argument("--limit", type=int, default=None, help="cap the questions planned")
    run_cmd.add_argument(
        "--parallel",
        type=int,
        default=None,
        help=(
            "how many runs may execute at once (default: the number of distinct systems, "
            "which is as much parallelism as the same-system rule can use)"
        ),
    )
    run_cmd.add_argument(
        "--allow-concurrent-same-system",
        action="store_true",
        help=(
            "let runs share one system; contaminates latency and rank stability, so "
            "those rows are marked contended and their system cost is withheld"
        ),
    )
    run_cmd.add_argument(
        "--contended",
        action="store_true",
        help="this run shares its system with a sibling (set by a batch parent)",
    )
    _add_paths(run_cmd)

    re_cmd = sub.add_parser("rescore", help="recompute report.json from existing artifacts")
    re_cmd.add_argument("run_id", help="run identifier under the artifacts root")
    re_cmd.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)

    ver = sub.add_parser("verify", help="replay the adversarial checks over existing artifacts")
    ver.add_argument("run_id", help="run identifier under the artifacts root")
    ver.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    ver.add_argument(
        "--metered",
        action="store_true",
        help="the run went through a gateway, so absent counters mean a bypass",
    )
    ver.add_argument(
        "--contended",
        action="store_true",
        help="the run shared its system, so cost attribution is unreliable",
    )

    ps_cmd = sub.add_parser("ps", help="list runs and their progress")
    ps_cmd.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    ps_cmd.add_argument("--all", action="store_true", help="include finished runs")
    ps_cmd.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ps_cmd.add_argument(
        "--watch", action="store_true", help="clear and redraw until no run is live"
    )
    ps_cmd.add_argument(
        "--interval", type=float, default=2.0, help="seconds between redraws (default: 2)"
    )

    for name, help_text in (
        ("logs", "show one run's status and the tail of its log"),
        ("stop", "signal a running run to stop"),
        ("rm", "delete a finished run's artifacts"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("run_id", help="run identifier under the artifacts root")
        cmd.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
        if name == "logs":
            cmd.add_argument(
                "--tail",
                type=int,
                default=DEFAULT_TAIL,
                help=f"log lines to show, or 0 for all (default: {DEFAULT_TAIL})",
            )
            cmd.add_argument(
                "-f",
                "--follow",
                action="store_true",
                help="keep printing new lines until the run stops; Ctrl-C stops watching",
            )

    return parser


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--datasets", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--repo-root", type=Path, default=Path())
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch; return a process exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "systems":
        return systems(args.registry, args.json)
    if args.command == "doctor":
        return doctor(args.system, args.registry, args.run_id)
    if args.command == "gateway":
        if args.gateway_command == "stop":
            return gateway_stop(compose_file=args.compose)
        if args.gateway_command == "status":
            return gateway_status(url=args.url, timeout=args.timeout)
        return gateway_start(
            compose_file=args.compose,
            config_file=args.config,
            url=args.url,
            wait=not args.no_wait,
            timeout=args.timeout,
            skip_env_check=args.skip_env_check,
        )
    if args.command == "run":
        return run(
            system=args.system,
            dataset=args.dataset,
            suite=args.suite,
            artifacts=args.artifacts,
            datasets=args.datasets,
            repo_root=args.repo_root,
            registry=args.registry,
            limit=args.limit,
            parallel=args.parallel,
            allow_concurrent_same_system=args.allow_concurrent_same_system,
            contended=args.contended,
            resume=args.resume,
            name=args.name,
            run_id_override=args.run_id,
            foreground=args.foreground,
        )
    if args.command == "ps":
        return ps(args.artifacts, args.all, args.json, args.watch, args.interval)

    # Every remaining command takes a run id, and every one accepts an unambiguous
    # prefix. Resolving here rather than in each command keeps one definition of what
    # "8c1e" means, and an ambiguous prefix lists the candidates instead of guessing.
    try:
        run_id = resolve_prefix(args.run_id, args.artifacts)
    except IdError as err:
        print(f"{err}", file=sys.stderr)
        return 2

    if args.command == "rescore":
        return rescore(run_id, args.artifacts)
    if args.command == "verify":
        return verify(run_id, args.artifacts, args.metered, args.contended)
    if args.command == "logs":
        return logs(run_id, args.artifacts, tail=args.tail, follow=args.follow)
    if args.command == "stop":
        return stop(run_id, args.artifacts)
    if args.command == "rm":
        return rm(run_id, args.artifacts)
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
