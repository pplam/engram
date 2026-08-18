"""`score` — a pure offline function from artifacts to `report.json` (ARCH §5, §7).

No clock, no randomness, no network. Same artifacts in, byte-identical report out,
which is what lets a third party recompute every published number from the bundle.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from orchestrator.artifacts import JsonlArtifact, read_json, write_json
from orchestrator.plan_records import ChunkRecord, QuestionRecord, parse_plan_record
from orchestrator.report import (
    Attribution,
    Completeness,
    Cost,
    Latency,
    Quality,
    Report,
    ReportRow,
    Robustness,
)
from orchestrator.report import RetrievalQuality as RetrievalSection
from orchestrator.retrieval_records import memories_of
from scorers.quality import get_scorer
from scorers.retrieval import score_retrieval

CROSS_DATASET_NOTE = (
    "Quality is comparable within this dataset only: datasets may pin their own judge, "
    "so averaging quality across datasets is not meaningful."
)
SYSTEM_COST_NOTE = (
    "System-side cost is unavailable: this run had no metering gateway, so only "
    "harness-side cost is reported."
)
UNRELIABLE_COST_NOTE = (
    "System-side per-phase cost is omitted: the ingest/retrieve boundary it is derived "
    "from was not clean for this run, so no defensible number exists (D6)."
)


def _wall_clock(rows: Iterable[dict[str, Any]], latencies: list[float]) -> int | None:
    """Return elapsed ms across `rows`, spanning concurrent work rather than summing it.

    Adds run concurrently under the suite's worker cap, so summing per-chunk latencies
    reports several times the time that actually passed. Archived bundles predating the
    timestamps fall back to the sum, which is what they were published with.
    """
    starts = [int(r["started_at_ms"]) for r in rows if "started_at_ms" in r]
    ends = [int(r["ended_at_ms"]) for r in rows if "ended_at_ms" in r]
    if starts and ends:
        return max(ends) - min(starts)
    return int(sum(latencies)) if latencies else None


def percentile(values: list[float], fraction: float) -> float | None:
    """Return the linear-interpolated percentile, or None for no samples."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _system_cost(
    attribution: Attribution, prompt: int | None, completion: int | None
) -> dict[str, Any]:
    """Return the system-cost fields, omitting numbers that are not defensible.

    Only a `derived` attribution carries figures. `unreliable` means the phase boundary
    D6(a) derives them from was broken, and `unavailable` means there was no gateway to
    derive from; in both cases the honest report is no number, not a plausible one.
    """
    if attribution != "derived":
        return {"system_attribution": attribution}
    total = None if prompt is None and completion is None else (prompt or 0) + (completion or 0)
    return {
        "system_attribution": attribution,
        "system_prompt_tokens": prompt,
        "system_completion_tokens": completion,
        "system_total_tokens": total,
    }


def _by_id(artifact: JsonlArtifact) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in artifact.stream()}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def run_score(
    run_dir: Path,
    in_process: bool = False,
    contended: bool = False,
    row_withheld: bool = False,
    system_attribution: Attribution = "unavailable",
    system_prompt_tokens: int | None = None,
    system_completion_tokens: int | None = None,
) -> Report:
    """Compute `report.json` from the artifacts in `run_dir` and return it."""
    manifest = read_json(run_dir / "manifest.json")
    dataset = manifest["dataset"]
    models = manifest["models"]
    top_k = int(manifest["top_k"])

    chunks: list[ChunkRecord] = []
    questions: dict[str, QuestionRecord] = {}
    for raw in JsonlArtifact(run_dir / "plan.jsonl").stream():
        record = parse_plan_record(raw)
        if isinstance(record, ChunkRecord):
            chunks.append(record)
        else:
            questions[record.id] = record

    ingested = _by_id(JsonlArtifact(run_dir / "ingest.jsonl"))
    retrieved = _by_id(JsonlArtifact(run_dir / "retrieve.jsonl"))
    answered = _by_id(JsonlArtifact(run_dir / "answer.jsonl"))
    judged = _by_id(JsonlArtifact(run_dir / "judge.jsonl"))

    scorer_name = str(dataset["scorer"])
    scorer = get_scorer(scorer_name)

    correct = 0
    scored = 0
    unparseable = 0
    used_judge = False
    exact_matches: list[float] = []
    token_f1s: list[float] = []
    rouge_ls: list[float] = []

    recalls: list[float] = []
    precisions: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    tracks: set[str] = set()
    correct_context_tokens: list[int] = []

    for qid, question in questions.items():
        answer_row = answered.get(qid)
        if answer_row is None:
            continue
        judge_row = judged.get(qid)
        if scorer_name == "judge_binary" and judge_row is None:
            continue

        label = str(judge_row["label"]) if judge_row else None
        if label == "UNPARSEABLE":
            unparseable += 1

        outcome = scorer(
            prediction=str(answer_row.get("generated_answer", "")),
            gold=tuple(question.gold),
            options=tuple(question.options) if question.options else None,
            judge_label=label,
        )
        scored += 1
        used_judge = used_judge or outcome.used_judge
        correct += int(outcome.is_correct)
        for values, value in (
            (exact_matches, outcome.exact_match),
            (token_f1s, outcome.token_f1),
            (rouge_ls, outcome.rouge_l),
        ):
            if value is not None:
                values.append(value)

        retrieve_row = retrieved.get(qid)
        if retrieve_row is not None:
            retrieval = score_retrieval(
                memories_of(retrieve_row),
                gold=tuple(question.evidence_chunks),
                k=top_k,
            )
            tracks.add(retrieval.track)
            for values, value in (
                (recalls, retrieval.recall_at_k),
                (precisions, retrieval.precision_at_k),
                (mrrs, retrieval.mrr),
                (ndcgs, retrieval.ndcg_at_k),
            ):
                if value is not None:
                    values.append(value)
            if outcome.is_correct:
                correct_context_tokens.append(retrieval.content_tokens)

    ingest_latencies = [float(r["latency_ms"]) for r in ingested.values() if "latency_ms" in r]
    retrieve_latencies = [float(r["latency_ms"]) for r in retrieved.values() if "latency_ms" in r]
    ingest_wall_clock = _wall_clock(ingested.values(), ingest_latencies)

    harness_prompt = sum(int(r.get("prompt_tokens", 0)) for r in answered.values()) + sum(
        int(r.get("prompt_tokens", 0)) for r in judged.values()
    )
    harness_completion = sum(int(r.get("completion_tokens", 0)) for r in answered.values()) + sum(
        int(r.get("completion_tokens", 0)) for r in judged.values()
    )

    row = ReportRow(
        system=str(manifest["system"]["name"]),
        dataset=str(dataset["name"]),
        suite=str(manifest["suite"]["version"]),
        run_id=str(manifest["run_id"]),
        scorer=scorer_name,
        top_k=top_k,
        limit=manifest.get("limit"),
        in_process=in_process,
        contended=contended,
        row_withheld=row_withheld,
        quality=Quality(
            scorer=scorer_name,
            accuracy=(correct / scored) if scored else 0.0,
            questions=scored,
            correct=correct,
            judge=str(models["judge"]) if used_judge else None,
            judge_source=models.get("judge_source") if used_judge else None,
            unparseable_judgements=unparseable,
            exact_match=_mean(exact_matches),
            token_f1=_mean(token_f1s),
            rouge_l=_mean(rouge_ls),
        ),
        retrieval=RetrievalSection(
            track="exact" if "exact" in tracks else "unavailable",
            evidence_unit=dataset["evidence_granularity"],
            measured_questions=len(recalls),
            recall_at_k=_mean(recalls),
            precision_at_k=_mean(precisions),
            mrr=_mean(mrrs),
            ndcg_at_k=_mean(ndcgs),
            context_tokens_per_correct=_mean([float(t) for t in correct_context_tokens]),
        ),
        cost=Cost(
            harness_attribution="measured",
            harness_prompt_tokens=harness_prompt,
            harness_completion_tokens=harness_completion,
            harness_llm_calls=len(answered) + len(judged),
            **_system_cost(system_attribution, system_prompt_tokens, system_completion_tokens),
        ),
        latency=Latency(
            ingest_p50_ms=percentile(ingest_latencies, 0.50),
            ingest_p95_ms=percentile(ingest_latencies, 0.95),
            ingest_p99_ms=percentile(ingest_latencies, 0.99),
            ingest_wall_clock_ms=ingest_wall_clock,
            retrieve_p50_ms=percentile(retrieve_latencies, 0.50),
            retrieve_p95_ms=percentile(retrieve_latencies, 0.95),
            retrieve_p99_ms=percentile(retrieve_latencies, 0.99),
        ),
        robustness=Robustness(
            ingest_retries=sum(max(int(r.get("attempts", 1)) - 1, 0) for r in ingested.values()),
            retrieve_retries=sum(max(int(r.get("attempts", 1)) - 1, 0) for r in retrieved.values()),
            ingest_failures=sum(1 for r in ingested.values() if r.get("status") != "ok"),
        ),
        completeness=Completeness(
            planned_chunks=len(chunks),
            planned_questions=len(questions),
            ingested_chunks=len(ingested),
            retrieved_questions=len(retrieved),
            answered_questions=len(answered),
            judged_questions=len(judged),
        ),
    )

    notes = [CROSS_DATASET_NOTE]
    if system_attribution == "unavailable":
        notes.append(SYSTEM_COST_NOTE)
    elif system_attribution == "unreliable":
        notes.append(UNRELIABLE_COST_NOTE)
    if in_process:
        notes.append(
            "This row is an in-process baseline: retrieval was injected rather than "
            "served over the contract, so it is exempt from the conformance suite."
        )
    if contended:
        notes.append(
            "This run shared its system with another concurrent run: latency and rank "
            "stability are contended and per-phase system cost is omitted."
        )
    if row_withheld:
        notes.append(
            "This row is withheld: verify.json records a check that could not be "
            "reconciled, so these numbers are not publishable. See verify.json."
        )

    report = Report(row=row, notes=notes)
    write_json(run_dir / "report.json", report.model_dump(mode="json"))
    return report
