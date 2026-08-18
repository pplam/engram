"""`answer` — generate an answer from retrieved content with the pinned model (ARCH §5).

The system returns memories; the harness writes the answer. That is what makes
answer generation a constant instead of a per-system confound.
"""

import time
from pathlib import Path

from orchestrator.artifacts import JsonlArtifact
from orchestrator.models import ChatClient
from orchestrator.plan_records import QuestionRecord, parse_plan_record
from orchestrator.prompting import render
from orchestrator.retrieval_records import MemoryRow, memories_of
from orchestrator.runlog import get_logger

log = get_logger("answer")


def format_memories(memories: list[MemoryRow]) -> str:
    """Render retrieved memories in the order the system ranked them."""
    if not memories:
        return "(no memories retrieved)"
    return "\n".join(f"- {item.get('content', '')}" for item in memories)


def _questions(run_dir: Path) -> dict[str, QuestionRecord]:
    plan = JsonlArtifact(run_dir / "plan.jsonl")
    records = (parse_plan_record(raw) for raw in plan.stream())
    return {r.id: r for r in records if isinstance(r, QuestionRecord)}


async def run_answer(
    run_dir: Path,
    chat: ChatClient,
    model: str,
    temperature: float,
    prompt: str,
    seed: int | None = None,
) -> int:
    """Answer every not-yet-answered retrieval; return how many were written."""
    retrieved = JsonlArtifact(run_dir / "retrieve.jsonl")
    out = JsonlArtifact(run_dir / "answer.jsonl")
    done = out.read_ids()
    questions = _questions(run_dir)
    written = 0
    if done:
        log.info("resuming: %d answers already present", len(done))

    for row in retrieved.stream():
        record_id = str(row["id"])
        if record_id in done:
            continue
        question = questions[record_id]
        memories = memories_of(row)
        rendered = render(
            prompt,
            memories=format_memories(memories),
            question=question.question,
        )
        started = time.monotonic()
        completion = await chat.chat(
            [{"role": "user", "content": rendered}],
            model=model,
            temperature=temperature,
            seed=seed,
        )
        # The id, the model, the sizes, and the latency. Never the prompt and never the
        # completion. The model is here rather than only on the stage boundary so a single
        # slow or refusing call can be attributed after the fact.
        log.info(
            "answer id=%s model=%s memories=%d latency_ms=%d prompt_tokens=%d completion_tokens=%d",
            record_id,
            model,
            len(memories),
            int((time.monotonic() - started) * 1000),
            completion.prompt_tokens,
            completion.completion_tokens,
        )
        out.append(
            {
                "id": record_id,
                "user_id": question.user_id,
                "generated_answer": completion.text,
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
            }
        )
        written += 1

    return written
