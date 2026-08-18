"""One run: six stages in strict order (ARCH §5, D6).

Stages are strictly sequential *by construction* here. That is not tidiness — the
system-side per-phase cost in Phase 3 is derived from the ingest/retrieve boundary,
so overlapping phases would silently corrupt a published number.

The two in-process baselines (D2) skip `ingest`/`retrieve` and inject retrieval
instead. That branch lives here at the run level, never inside a stage.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

from contract.adapter import MemoryAdapter
from orchestrator.artifacts import ArtifactError, read_json
from orchestrator.client import AdapterClient, RetryPolicy
from orchestrator.datasets import load_dataset
from orchestrator.gateway import Gateway, GatewayError, PinnedModel, RunKeys, wire_name
from orchestrator.metrics import Snapshot, Usage, diff
from orchestrator.models import ChatClient
from orchestrator.progress import Progress
from orchestrator.report import Report
from orchestrator.resolution import resolve_answer_prompt, resolve_judge
from orchestrator.runlog import configure, get_logger
from orchestrator.stages.answer import run_answer
from orchestrator.stages.ingest import run_ingest
from orchestrator.stages.judge import run_judge
from orchestrator.stages.liveness import run_liveness
from orchestrator.stages.plan import run_plan
from orchestrator.stages.retrieve import run_retrieve
from orchestrator.stages.score import run_score
from orchestrator.suite import load_suite
from orchestrator.verify import run_verify
from reference.baselines.in_process import Baseline, inject_retrieval

IN_PROCESS: tuple[str, ...] = ("oracle_gold", "long_context")

# The stage sequence, in execution order, as `bench ps` numbers it. The in-process
# baselines inject retrieval instead of ingesting (D2), so they genuinely run six stages
# and must not be reported as stuck at "2/7".
STAGES: tuple[str, ...] = (
    "plan",
    "ingest",
    "retrieve",
    "answer",
    "judge",
    "verify",
    "score",
)
IN_PROCESS_STAGES: tuple[str, ...] = tuple(s for s in STAGES if s != "ingest")

# Printed when key provisioning fails. The run is still valid and still publishes a row —
# only its cost column is empty — so this has to be visible without being an error, or an
# unmetered run reads as a metered one until someone goes looking for the number.
UNMETERED_HINT = (
    "gateway did not provision keys, so this run is unmetered and its cost column "
    "will read `unavailable`. Start it with `bench gateway start`"
)

log = get_logger("run")


class RunError(Exception):
    """The run could not be executed as requested."""


@dataclass(frozen=True)
class ResumedConfig:
    """The configuration a run was started with, read back from its own manifest."""

    system: str
    dataset: str
    suite: str
    limit: int | None
    name: str | None


def _manifest_of(run_dir: Path) -> dict[str, object]:
    """Return `run_dir`'s manifest, or raise RunError explaining why it cannot be read."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RunError(f"cannot resume {run_dir.name}: no manifest.json under {run_dir}")
    try:
        return read_json(manifest_path)
    except (ArtifactError, OSError) as err:
        raise RunError(f"cannot resume {run_dir.name}: manifest.json is unreadable: {err}") from err


def resumed_config(run_dir: Path) -> ResumedConfig:
    """Return the configuration recorded in `run_dir`'s manifest (§4.5).

    A run's manifest already says what the run *is*, so `--resume <id>` does not need the
    system, dataset, suite, or limit retyped: the only thing a repeated argument can do is
    match this or be refused. Reading it back also closes the gap where an *omitted*
    `--limit` replanned an existing run at a different size.

    A missing field is refused rather than defaulted: guessing a blank system or dataset
    would run a configuration nobody asked for.
    """
    manifest = _manifest_of(run_dir)
    limit = manifest.get("limit")
    return ResumedConfig(
        system=_required(run_dir, manifest, "system", "name"),
        dataset=_required(run_dir, manifest, "dataset", "name"),
        suite=_required(run_dir, manifest, "suite", "version"),
        limit=int(limit) if isinstance(limit, int) else None,
        # A label, not part of the configuration, so an absent one is simply no label.
        name=str(manifest["name"]) if isinstance(manifest.get("name"), str) else None,
    )


def _recorded(manifest: dict[str, object], section: str, field: str) -> str:
    """Return `manifest[section][field]` as a string, or "" when it is absent."""
    block = manifest.get(section)
    value = block.get(field) if isinstance(block, dict) else None
    return str(value) if value is not None else ""


def _required(run_dir: Path, manifest: dict[str, object], section: str, field: str) -> str:
    """Return `manifest[section][field]`, raising RunError when it is absent or empty."""
    block = manifest.get(section)
    value = block.get(field) if isinstance(block, dict) else None
    if not isinstance(value, str) or not value:
        raise RunError(
            f"cannot resume {run_dir.name}: its manifest.json records no "
            f"{section} {field}, so the configuration to resume is unknown"
        )
    return value


def check_resumable(
    run_dir: Path, system: str, dataset: str, suite: str, limit: int | None = None
) -> None:
    """Raise RunError unless `run_dir` was produced by this same configuration (§4.5).

    Artifacts are append-only, so resuming is just "skip the IDs already present". That
    is only sound if the run being continued is the *same* run — otherwise one directory
    ends up holding rows from two configurations and one `report.json` averaging across
    them, which is precisely the splice that makes a published number indefensible.

    `limit` is checked for the same reason: it decides how much of the dataset was
    planned, so resuming under a different one replans the run the existing artifacts
    were built against.
    """
    manifest = _manifest_of(run_dir)

    recorded_limit = manifest.get("limit")
    recorded_limit = recorded_limit if isinstance(recorded_limit, int) else None
    if recorded_limit != limit:
        raise RunError(
            f"cannot resume {run_dir.name}: it was planned with limit "
            f"{recorded_limit!r}, but {limit!r} was requested. Resuming would replan it "
            "at a different size than its artifacts were built for."
        )

    recorded = {
        "system": _recorded(manifest, "system", "name"),
        "dataset": _recorded(manifest, "dataset", "name"),
        "suite": _recorded(manifest, "suite", "version"),
    }
    for field, requested in (("system", system), ("dataset", dataset), ("suite", suite)):
        if recorded[field] != requested:
            raise RunError(
                f"cannot resume {run_dir.name}: it was run with {field} "
                f"{recorded[field]!r}, but {requested!r} was requested. Resuming would "
                "splice two configurations into one report."
            )


@dataclass(frozen=True)
class RunRequest:
    """Everything a single run needs. No per-run tuning knobs by design (ARCH §11)."""

    run_id: str
    system: str
    dataset: str
    suite: str
    artifacts_root: Path
    datasets_root: Path
    repo_root: Path
    limit: int | None = None
    contended: bool = False
    # A human label for `bench ps`. Metadata, not the namespace — `run_id` is the
    # namespace — so it need not be unique and nothing keys off it.
    name: str | None = None

    @property
    def run_dir(self) -> Path:
        """The per-run artifact directory."""
        return self.artifacts_root / self.run_id

    @property
    def is_in_process(self) -> bool:
        """True for the two privileged baselines that bypass the contract (D2)."""
        return self.system in IN_PROCESS


async def execute_run(
    request: RunRequest,
    adapter: MemoryAdapter | None,
    chat: ChatClient,
    judge_chat: ChatClient,
    gateway: Gateway | None = None,
) -> Report:
    """Run all seven stages in order and return the report.

    With a `gateway`, `/metrics` is snapshotted at each phase boundary so per-phase cost
    is a counter diff (D6). Without one, model calls go direct and `verify` records
    metering as unmeasurable rather than as a bypass.
    """
    progress = Progress(
        request.run_dir,
        run_id=request.run_id,
        system=request.system,
        dataset=request.dataset,
        suite=request.suite,
        stages=IN_PROCESS_STAGES if request.is_in_process else STAGES,
        name=request.name,
    )
    # The log is opened before the first stage, so a run that dies in `plan` still says
    # why. It lives beside the artifacts because that directory is the run's identity.
    configure(request.run_dir)
    log.info(
        "run start run_id=%s system=%s dataset=%s suite=%s limit=%s contended=%s metered=%s",
        request.run_id,
        request.system,
        request.dataset,
        request.suite,
        request.limit,
        request.contended,
        gateway is not None,
    )
    started = time.monotonic()
    try:
        with progress:
            report = await _execute(request, adapter, chat, judge_chat, gateway, progress)
    except BaseException as err:
        # Type and message only — an exception carrying a payload must not put one in
        # the log. Re-raised unchanged: the caller still decides what a failure means.
        log.error(
            "run failed run_id=%s elapsed_ms=%d error=%s: %s",
            request.run_id,
            int((time.monotonic() - started) * 1000),
            type(err).__name__,
            err,
        )
        raise
    log.info(
        "run done run_id=%s elapsed_ms=%d", request.run_id, int((time.monotonic() - started) * 1000)
    )
    return report


async def _execute(
    request: RunRequest,
    adapter: MemoryAdapter | None,
    chat: ChatClient,
    judge_chat: ChatClient,
    gateway: Gateway | None,
    progress: Progress,
) -> Report:
    """The seven stages themselves, with progress recorded around each."""
    suite = load_suite(request.suite, request.repo_root)
    dataset = load_dataset(request.dataset, request.datasets_root)

    # Logged as its own line rather than on `run start`, which is written before the suite
    # is loaded and so cannot know them. The *resolved* judge, because a dataset may pin
    # its own and override the suite (D11) — the suite's id would name a model that never
    # ran. Provider-prefixed, since a pinned model is `(provider, id)` and the id alone
    # does not say which endpoint answered.
    # Resolved once for the whole run: a dataset may override the suite's judge (D11), and
    # two resolution sites is two places for the logged judge and the running one to drift.
    judge = resolve_judge(suite, dataset.config)
    log.info(
        "run models chat_model=%s judge_model=%s judge_source=%s embedding_model=%s",
        wire_name(PinnedModel(suite.models.chat.id, suite.models.chat.provider)),
        wire_name(PinnedModel(judge.id, judge.provider)),
        judge.source,
        wire_name(PinnedModel(suite.models.embedding.id, suite.models.embedding.provider)),
    )

    keys: RunKeys | None = None
    if gateway:
        try:
            # Pinning is enforced by the gateway: a model outside this list gets a 403
            # rather than relying on anyone honouring the suite's text.
            keys = await gateway.provision(
                request.run_id,
                request.system,
                allowed_models=[
                    PinnedModel(suite.models.chat.id, suite.models.chat.provider),
                    PinnedModel(suite.models.embedding.id, suite.models.embedding.provider),
                    PinnedModel(judge.id, judge.provider),
                ],
            )
        except GatewayError as err:
            # Unmetered is a supported configuration, not a failure: a gateway may be down
            # or have governance disabled, and an in-repo baseline needs none at all.
            # Dropping the gateway here rather than half-metering is what keeps the cost
            # column honest — every snapshot below is guarded, and `metered` goes false, so
            # the row reports `unavailable` instead of a zero nobody could defend.
            log.warning("gateway did not provision keys, running unmetered: %s", err)
            print(f"warning: {UNMETERED_HINT} ({err})", file=sys.stderr)
            gateway = None

    if gateway and keys:
        # Each half of the run presents the key that labels its own traffic. Handed over
        # here, before any stage runs, because a call made before its key arrives is a
        # call no counter can attribute: the gateway labels a series by the key it saw.
        await _present_key(chat, keys.harness["answer"].value, request.run_id, "answer")
        await _present_key(judge_chat, keys.harness["judge"].value, request.run_id, "judge")

        # The system gets one run-scoped key (D6(a)) — a black box cannot take a
        # phase-scoped one, so its per-phase split comes from the snapshot boundary.
        # Optional on the protocol, like `close`: most systems cannot be handed a
        # credential at runtime, and the ones that cannot simply go unmetered.
        deliver = getattr(adapter, "use_model_key", None)
        if deliver is not None:
            await deliver(keys.system.value)
            log.info("system key delivered vk_id=%s", keys.system.vk_id)

    progress.enter("plan")
    log.info("plan start dataset=%s limit=%s", dataset.config.name, request.limit)
    planned = run_plan(
        run_id=request.run_id,
        system=request.system,
        dataset=dataset,
        suite=suite,
        out_dir=request.run_dir,
        limit=request.limit,
        name=request.name,
    )
    log.info("plan done chunks=%d questions=%d", planned.chunks, planned.questions)

    usage: dict[str, Usage] = {}
    top_k = suite.retrieval.top_k
    if request.is_in_process:
        if adapter is not None:
            raise RunError(f"{request.system!r} is an in-process baseline and takes no adapter")
        progress.enter("retrieve")
        log.info("retrieve start in_process=%s top_k=%d", request.system, top_k)
        # Injection is synchronous and local, so there is nothing to tick through: it
        # cannot be observed part-done. The count is written once, after.
        inject_retrieval(request.run_dir, baseline=_baseline(request.system), top_k=top_k)
        progress.beat("retrieve.jsonl", planned.questions, unit="searches")
        log.info("retrieve done injected=%d", planned.questions)
    else:
        if adapter is None:
            raise RunError(f"{request.system!r} needs an adapter")
        try:
            before_ingest = await gateway.snapshot() if gateway else None
            # One AdapterClient per phase: each phase has its own worker bound and its
            # own retry policy, and the adapter itself is shared and closed once.
            ingest_client = AdapterClient(adapter, RetryPolicy.for_stage(suite.limits.ingest))
            progress.enter("ingest")
            log.info(
                "ingest start chunks=%d workers=%d timeout_s=%d",
                planned.chunks,
                suite.limits.ingest.workers,
                suite.limits.ingest.timeout_s,
            )
            # Ticking, not a count afterwards: ingest is the longest stage, and a number
            # written only when it returns is a number that never moved while it ran.
            async with progress.ticking("ingest.jsonl", planned.chunks, unit="chunks"):
                ingested = await run_ingest(request.run_dir, ingest_client)
            log.info("ingest done written=%d of chunks=%d", ingested, planned.chunks)

            # Ingest is fully drained before retrieve opens. This boundary is what makes
            # system-side per-phase cost attributable at all (D6) — never overlap them.
            # The system holds one run-scoped key, so its per-phase split is exactly the
            # counter delta on either side of this line and nothing else.
            if gateway and before_ingest is not None:
                usage["ingest"] = _system_usage(before_ingest, await gateway.snapshot(), keys)

            # Between the two snapshot windows on purpose: the gate issues searches, and
            # counting them as ingest would inflate write cost per chunk while counting
            # them as retrieve would inflate read cost per query. Neither published
            # per-phase number may move because we checked.
            probed = await run_liveness(
                request.run_dir,
                ingest_client,
                top_k=top_k,
                # Read off the adapter, like `close`: only the system can declare that
                # an empty store is its measurement rather than a dropped write.
                stores_nothing=bool(getattr(adapter, "stores_nothing", False)),
                ingested=ingested,
            )
            log.info("liveness done contexts_probed=%d", probed)

            before_retrieve = await gateway.snapshot() if gateway else None
            retrieve_client = AdapterClient(adapter, RetryPolicy.for_stage(suite.limits.retrieve))
            progress.enter("retrieve")
            log.info(
                "retrieve start questions=%d top_k=%d workers=%d",
                planned.questions,
                top_k,
                suite.limits.retrieve.workers,
            )
            async with progress.ticking("retrieve.jsonl", planned.questions, unit="searches"):
                searched = await run_retrieve(request.run_dir, retrieve_client, top_k=top_k)
            log.info("retrieve done written=%d of questions=%d", searched, planned.questions)
            if gateway and before_retrieve is not None:
                usage["retrieve"] = _system_usage(before_retrieve, await gateway.snapshot(), keys)
        finally:
            closer = getattr(adapter, "close", None)
            if closer is not None:
                await closer()

    progress.enter("answer")
    before_answer = await gateway.snapshot() if gateway else None
    answer_prompt = resolve_answer_prompt(suite, dataset.config)
    log.info(
        "answer start questions=%d model=%s temperature=%s prompt_source=%s",
        planned.questions,
        wire_name(PinnedModel(suite.models.chat.id, suite.models.chat.provider)),
        suite.models.chat.temperature,
        answer_prompt.source,
    )
    async with progress.ticking("answer.jsonl", planned.questions, unit="answers"):
        answered = await run_answer(
            request.run_dir,
            chat,
            # Provider-prefixed: the gateway resolves `<provider>/<id>`, and an id
            # containing a slash is otherwise read as naming a provider that does not exist.
            model=wire_name(PinnedModel(suite.models.chat.id, suite.models.chat.provider)),
            temperature=suite.models.chat.temperature,
            prompt=answer_prompt.text,
            seed=suite.models.chat.seed,
        )

    if gateway and before_answer is not None:
        usage["answer"] = _phase_usage(before_answer, await gateway.snapshot(), keys, "answer")

    log.info("answer done written=%d of questions=%d", answered, planned.questions)

    progress.enter("judge")
    before_judge = await gateway.snapshot() if gateway else None
    log.info(
        "judge start answers=%d model=%s source=%s",
        answered,
        wire_name(PinnedModel(judge.id, judge.provider)),
        judge.source,
    )
    async with progress.ticking("judge.jsonl", planned.questions, unit="verdicts"):
        judged = await run_judge(
            request.run_dir,
            judge_chat,
            model=wire_name(PinnedModel(judge.id, judge.provider)),
            temperature=judge.temperature,
            prompt=suite.prompts.judge.text,
        )
    if gateway and before_judge is not None:
        usage["judge"] = _phase_usage(before_judge, await gateway.snapshot(), keys, "judge")

    log.info("judge done written=%d", judged)

    progress.enter("verify")
    log.info("verify start")
    # `verify` runs before `score` so the report can carry its outcome. Both are offline
    # and touch no memory system, which is what lets a reviewer recompute them (P4).
    verification = run_verify(
        request.run_dir,
        usage=usage,
        contended=request.contended,
        metered=gateway is not None,
    )
    log.info(
        "verify done run_valid=%s row_withheld=%s cost_attribution=%s flags=%s",
        verification.run_valid,
        verification.row_withheld,
        verification.cost_attribution,
        ",".join(verification.flags) or "-",
    )

    # The system's cost is the sum of its two phases, and it is published only if
    # `verify` concluded the boundary it was derived from is sound.
    system = Usage(
        prompt_tokens=sum(usage.get(p, Usage()).prompt_tokens for p in ("ingest", "retrieve")),
        completion_tokens=sum(
            usage.get(p, Usage()).completion_tokens for p in ("ingest", "retrieve")
        ),
    )
    derived = verification.cost_attribution == "derived"

    progress.enter("score")
    log.info("score start derived_system_cost=%s", derived)
    report = run_score(
        request.run_dir,
        in_process=request.is_in_process,
        contended=request.contended,
        row_withheld=verification.row_withheld,
        system_attribution=verification.cost_attribution,
        system_prompt_tokens=system.prompt_tokens if derived else None,
        system_completion_tokens=system.completion_tokens if derived else None,
    )
    log.info(
        "score done accuracy=%.4f correct=%d of questions=%d withheld=%s",
        report.row.quality.accuracy,
        report.row.quality.correct,
        report.row.quality.questions,
        report.row.row_withheld,
    )
    return report


async def _present_key(client: ChatClient, value: str, run_id: str, phase: str) -> None:
    """Hand one phase's virtual key to its chat client, if the client can take one.

    Optional for the same reason `close` is: a stub has no headers to set, and requiring
    the method would put a no-op on every implementation of the protocol.
    """
    present = getattr(client, "use_key", None)
    if present is not None:
        await present(value, run_id, phase)


def _phase_usage(before: Snapshot, after: Snapshot, keys: RunKeys | None, phase: str) -> Usage:
    """Return what one harness phase spent, from the counter diff for its own key (D3)."""
    key = keys.harness.get(phase) if keys else None
    if key is None:
        return Usage()
    return diff(before, after).get(key.vk_id, Usage())


def _system_usage(before: Snapshot, after: Snapshot, keys: RunKeys | None) -> Usage:
    """Return what the system spent in one phase, off its single run-scoped key (D6(a))."""
    if keys is None:
        return Usage()
    return diff(before, after).get(keys.system.vk_id, Usage())


def _baseline(name: str) -> Baseline:
    if name == "oracle_gold":
        return "oracle_gold"
    if name == "long_context":
        return "long_context"
    raise RunError(f"{name!r} is not an in-process baseline")
