"""`plan` — dataset + suite in, `plan.jsonl` + `manifest.json` out (ARCH §5).

Deterministic: same dataset and suite produce a byte-identical plan. The manifest
is written before anything else, so a run that cannot record its pins never starts.
`--limit` truncates here, and every downstream stage inherits it for free.
"""

from dataclasses import dataclass
from pathlib import Path

from orchestrator.artifacts import JsonlArtifact, write_json
from orchestrator.chunking import Chunk, chunk_messages
from orchestrator.datasets import Context, Dataset, Question
from orchestrator.evidence import expand_evidence
from orchestrator.ids import question_id, request_id, user_id
from orchestrator.plan_records import ChunkRecord, PlanRecord, QuestionRecord
from orchestrator.resolution import resolve_answer_prompt, resolve_judge
from orchestrator.suite import Suite


class PlanError(Exception):
    """The plan could not be written."""


@dataclass(frozen=True)
class PlanOutcome:
    """What the plan stage produced."""

    chunks: int
    questions: int


def _limited(
    dataset: Dataset, limit: int | None
) -> tuple[tuple[Context, ...], tuple[Question, ...]]:
    """Return the questions kept under `limit`, plus only the contexts they need."""
    if limit is None:
        return dataset.contexts, dataset.questions
    questions = dataset.questions[:limit]
    needed = {q.ctx_id for q in questions}
    return tuple(c for c in dataset.contexts if c.ctx_id in needed), questions


def _manifest(
    run_id: str,
    system: str,
    dataset: Dataset,
    suite: Suite,
    limit: int | None,
    name: str | None = None,
) -> dict[str, object]:
    judge = resolve_judge(suite, dataset.config)
    answer_prompt = resolve_answer_prompt(suite, dataset.config)
    return {
        "run_id": run_id,
        # A human label, not an identifier: `run_id` is the namespace.
        "name": name,
        "system": {"name": system},
        "suite": {"version": suite.suite, "sha256": suite.sha256},
        "dataset": {
            "name": dataset.config.name,
            "version": dataset.config.version,
            "source_sha256": dataset.config.source.sha256,
            "scorer": dataset.config.scorer,
            "task_type": dataset.config.task_type,
            "evidence_granularity": dataset.config.evidence_granularity,
            "supports_recall": dataset.config.supports_recall,
        },
        "models": {
            "chat": suite.models.chat.id,
            "embedding": suite.models.embedding.id,
            "judge": judge.id,
            "judge_source": judge.source,
            # A pinned model is `(provider, id)` (D11), so the id alone does not say
            # which endpoint answered. Absent for a run under a legacy suite, which
            # named no provider to record.
            "providers": {
                "chat": suite.models.chat.provider,
                "embedding": suite.models.embedding.provider,
                "judge": judge.provider,
            },
        },
        "prompts": {
            "answer_sha256": answer_prompt.sha256,
            "answer_source": answer_prompt.source,
            "judge_sha256": suite.prompts.judge.sha256,
        },
        "top_k": suite.retrieval.top_k,
        "limit": limit,
    }


def _plan_records(
    run_id: str, dataset: Dataset, suite: Suite, limit: int | None
) -> list[PlanRecord]:
    contexts, questions = _limited(dataset, limit)
    name = dataset.config.name
    unit = dataset.config.evidence_granularity
    records: list[PlanRecord] = []
    chunked: dict[str, list[Chunk]] = {}

    for context in contexts:
        uid = user_id(run_id, name, context.ctx_id)
        chunks = chunk_messages(list(context.messages), suite.chunking)
        chunked[context.ctx_id] = chunks
        for chunk in chunks:
            records.append(
                ChunkRecord(
                    id=request_id(uid, chunk.chunk_id),
                    user_id=uid,
                    ctx_id=context.ctx_id,
                    chunk_id=chunk.chunk_id,
                    messages=list(chunk.messages),
                )
            )

    for question in questions:
        uid = user_id(run_id, name, question.ctx_id)
        # Gold evidence arrives in the dataset's unit; recall needs chunk ids (ARCH §7.2).
        expanded = expand_evidence(
            question.evidence_chunk_ids, chunked.get(question.ctx_id, []), unit
        )
        records.append(
            QuestionRecord(
                id=question_id(uid, question.q_id),
                user_id=uid,
                ctx_id=question.ctx_id,
                q_id=question.q_id,
                question=question.question,
                gold=list(question.gold),
                options=list(question.options) if question.options else None,
                evidence_chunk_ids=list(question.evidence_chunk_ids),
                evidence_chunks=list(expanded),
                meta=dict(question.meta),
            )
        )
    return records


def run_plan(
    run_id: str,
    system: str,
    dataset: Dataset,
    suite: Suite,
    out_dir: Path,
    limit: int | None = None,
    name: str | None = None,
) -> PlanOutcome:
    """Write `manifest.json` then `plan.jsonl`; return the record counts."""
    try:
        write_json(
            out_dir / "manifest.json",
            _manifest(run_id, system, dataset, suite, limit, name),
        )
    except OSError as err:
        raise PlanError(f"cannot write manifest.json under {out_dir}: {err}") from err

    artifact = JsonlArtifact(out_dir / "plan.jsonl")
    done = artifact.read_ids()
    records = _plan_records(run_id, dataset, suite, limit)
    for record in records:
        if record.id not in done:
            artifact.append(record.model_dump(exclude_none=True, mode="json"))

    return PlanOutcome(
        chunks=sum(1 for r in records if isinstance(r, ChunkRecord)),
        questions=sum(1 for r in records if isinstance(r, QuestionRecord)),
    )
