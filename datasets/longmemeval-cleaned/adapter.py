"""LongMemEval-S Cleaned: one private haystack per question, session-level evidence.

Upstream shape (`longmemeval_s_cleaned.json`, one entry per question):

    {"question_id": "e47becba", "question": "...", "answer": "...",
     "question_type": "single-session-user", "question_date": "...",
     "haystack_session_ids": ["sharegpt_yywfIrx_0", "answer_280352e9", ...],
     "haystack_sessions": [[{"role","content"}, ...], ...],
     "haystack_dates": ["...", ...],
     "answer_session_ids": ["answer_280352e9"]}

The exact opposite shape to LoCoMo: ~53 sessions of haystack belong to one question and
are never shared, where LoCoMo has ~138 questions over one conversation. Evidence is
session-granular, so one gold session expands to every chunk derived from it — a
systematically easier recall target, which is why the unit is labelled on the report.

Turns already use `role`/`content`, so no field renaming is needed. What this adapter does
add is `session_id` on every message: the harness chunks a flat message stream, and the
session a message came from is the only thing that lets a gold session id resolve to the
chunks that carry it.
"""

import json
from collections.abc import Iterable
from typing import Any

from orchestrator.datasets import Context, DatasetConfig, Question


def load(config: DatasetConfig) -> tuple[Iterable[Context], Iterable[Question]]:
    """Return one context per question, plus that question."""
    raw: Any = json.loads(config.source_path.read_text())

    contexts: list[Context] = []
    questions: list[Question] = []
    for index, entry in enumerate(raw):
        ctx_id = str(entry.get("question_id") or f"longmemeval_{index}")
        session_ids: list[Any] = entry.get("haystack_session_ids") or []
        sessions: list[Any] = entry.get("haystack_sessions") or []
        dates: list[Any] = entry.get("haystack_dates") or []

        messages: list[dict[str, Any]] = []
        for position, session in enumerate(sessions):
            # Upstream ids are what `answer_session_ids` refers to, so they are carried
            # through verbatim; renumbering them by position would break gold matching.
            session_id = (
                str(session_ids[position]) if position < len(session_ids) else f"session_{position}"
            )
            when = str(dates[position]) if position < len(dates) else ""
            for turn in session:
                messages.append(
                    {
                        "role": str(turn.get("role", "user")),
                        "content": str(turn.get("content", "")),
                        "session_id": session_id,
                        "timestamp": when,
                    }
                )
        contexts.append(Context(ctx_id=ctx_id, messages=tuple(messages)))

        questions.append(
            Question(
                q_id=ctx_id,
                ctx_id=ctx_id,
                question=str(entry.get("question", "")),
                gold=(str(entry.get("answer", "")),),
                evidence_chunk_ids=tuple(str(s) for s in entry.get("answer_session_ids") or ()),
                meta={
                    "question_type": str(entry.get("question_type", "")),
                    "question_date": str(entry.get("question_date", "")),
                },
            )
        )
    return contexts, questions
