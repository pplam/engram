"""LoCoMo-shaped fixture: many questions per conversation.

Evidence is annotated at message level: `m<i>` names the i-th message of the
conversation, which the loader maps onto chunk ids downstream.
"""

import json
from collections.abc import Iterable
from typing import Any

from orchestrator.datasets import Context, DatasetConfig, Question


def load(config: DatasetConfig) -> tuple[Iterable[Context], Iterable[Question]]:
    """Return contexts and questions parsed from the pinned corpus."""
    raw: dict[str, Any] = json.loads(config.source_path.read_text())

    contexts = [
        Context(ctx_id=conv["sample_id"], messages=tuple(conv["messages"]))
        for conv in raw["conversations"]
    ]
    questions = [
        Question(
            q_id=q["qa_id"],
            ctx_id=q["sample_id"],
            question=q["question"],
            gold=tuple(q["answer"]),
            options=tuple(q["options"]) if q.get("options") else None,
            evidence_chunk_ids=tuple(q.get("evidence_messages") or ()),
            meta={"category": q["category"]},
        )
        for q in raw["questions"]
    ]
    return contexts, questions
