"""LongMemEval-shaped fixture: one private haystack per question.

Evidence is annotated at session level, so a gold session expands to every chunk
derived from it. The adapter flattens sessions into one message stream and records
which session each message came from, so the mapping survives chunking.
"""

import json
from collections.abc import Iterable
from typing import Any

from orchestrator.datasets import Context, DatasetConfig, Question


def load(config: DatasetConfig) -> tuple[Iterable[Context], Iterable[Question]]:
    """Return one context per question, plus that question."""
    raw: dict[str, Any] = json.loads(config.source_path.read_text())

    contexts: list[Context] = []
    questions: list[Question] = []
    for instance in raw["instances"]:
        ctx_id = instance["question_id"]
        messages: list[dict[str, Any]] = []
        for index, session in enumerate(instance["haystack_sessions"]):
            for message in session:
                messages.append({**message, "session_id": f"sess{index}"})

        contexts.append(Context(ctx_id=ctx_id, messages=tuple(messages)))
        questions.append(
            Question(
                q_id=instance["question_id"],
                ctx_id=ctx_id,
                question=instance["question"],
                gold=(instance["answer"],),
                evidence_chunk_ids=tuple(instance["answer_session_ids"]),
                meta={"question_type": instance["question_type"]},
            )
        )
    return contexts, questions
