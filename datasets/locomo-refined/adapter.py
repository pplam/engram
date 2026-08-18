"""LoCoMo-Refined: many questions over one long multi-session conversation.

Upstream shape (`locomo10.json`, one entry per conversation):

    {"sample_id": "conv-26",
     "conversation": {"speaker_a": "...", "speaker_b": "...",
                      "session_1": [{"speaker","text","dia_id"}, ...],
                      "session_1_date_time": "...", ...},
     "qa": [{"question","answer","evidence":["D1:3"],"category":2}, ...]}

Two upstream quirks this handles, both verified against the real file:

- **Category 5 answers live in `adversarial_answer`.** These are questions whose honest
  answer is "not stated", and 444 of 446 have no `answer` key at all. Reading `answer`
  would silently score them against an empty string.
- **A few `evidence` entries pack several handles into one string** — `"D8:6; D9:17"` and
  three space-separated cases. Splitting recovers 4 of the 9 that would not otherwise
  resolve; the remaining 5 are upstream typos (`"D"`, `"D:11:26"`, `"D10:19"`, `"D4:36"`,
  `"D30:05"`) and are left alone deliberately. A handle that resolves to no message drops
  that question to the `unavailable` recall track, which is honest — inventing evidence to
  make it resolve would fabricate a measurement.

Evidence is message-granular: `dia_id` names one turn, and the orchestrator expands those
handles to chunk ids at plan time.
"""

import json
import re
from collections.abc import Iterable
from typing import Any

from orchestrator.datasets import Context, DatasetConfig, Question

# Handles are separated by whitespace, commas, or semicolons in the wild.
_SEPARATORS = re.compile(r"[;,\s]+")


def _handles(evidence: Any) -> tuple[str, ...]:
    """Return the evidence handles in one `evidence` list, unpacking multi-id strings."""
    if not isinstance(evidence, list):
        return ()
    out: list[str] = []
    for entry in evidence:
        for part in _SEPARATORS.split(str(entry).strip()):
            if part and part not in out:
                out.append(part)
    return tuple(out)


def _sessions(conversation: dict[str, Any]) -> list[str]:
    """Return the conversation's session keys in chronological order."""
    keys = [
        key
        for key, value in conversation.items()
        if key.startswith("session_") and not key.endswith("_date_time")
        if isinstance(value, list)
    ]
    # `session_10` must sort after `session_2`, so sort numerically rather than as text.
    return sorted(keys, key=lambda key: int(key.split("_")[1]))


def _speakers(conversation: dict[str, Any]) -> tuple[str, str]:
    """Return the two speakers, so turns can be mapped onto `user`/`assistant` roles."""
    return (
        str(conversation.get("speaker_a", "") or ""),
        str(conversation.get("speaker_b", "") or ""),
    )


def _content(turn: dict[str, Any], speaker: str) -> str:
    """Return the turn's text, prefixed by its speaker and including any image caption."""
    text = str(turn.get("text", "") or "").strip()
    caption = str(turn.get("blip_caption", "") or "").strip()
    if caption:
        # A described image is part of what was said; dropping it loses answerable detail.
        text = f"[Image: {caption}] {text}".strip()
    return f"{speaker}: {text}".strip()


def _answer(qa: dict[str, Any]) -> str:
    """Return the gold answer, preferring the adversarial one where upstream defines it."""
    if qa.get("adversarial_answer") is not None:
        return str(qa["adversarial_answer"])
    return str(qa.get("answer", ""))


def load(config: DatasetConfig) -> tuple[Iterable[Context], Iterable[Question]]:
    """Return one context per conversation, plus every question asked about it."""
    raw: Any = json.loads(config.source_path.read_text())

    contexts: list[Context] = []
    questions: list[Question] = []
    for index, sample in enumerate(raw):
        ctx_id = str(sample.get("sample_id") or f"locomo_{index}")
        conversation: dict[str, Any] = sample.get("conversation", {})

        speakers = _speakers(conversation)
        messages: list[dict[str, Any]] = []
        for key in _sessions(conversation):
            when = str(conversation.get(f"{key}_date_time", "") or "")
            for turn in conversation[key]:
                speaker = str(turn.get("speaker", "Unknown") or "Unknown")
                messages.append(
                    {
                        # `role` and `content` are the harness's contract; normalizing here
                        # is the adapter's job. The speaker's name stays inside `content`
                        # because who said a thing is answerable detail, and a two-party
                        # `role` cannot carry it.
                        "role": "user" if speaker == speakers[0] else "assistant",
                        "content": _content(turn, speaker),
                        "speaker": speaker,
                        "dia_id": str(turn.get("dia_id", "")),
                        "session_id": key,
                        "timestamp": when,
                    }
                )
        contexts.append(Context(ctx_id=ctx_id, messages=tuple(messages)))

        for position, qa in enumerate(sample.get("qa", [])):
            category = qa.get("category")
            questions.append(
                Question(
                    q_id=str(qa.get("question_id") or f"{ctx_id}_qa{position}"),
                    ctx_id=ctx_id,
                    question=str(qa.get("question", "")),
                    gold=(_answer(qa),),
                    evidence_chunk_ids=_handles(qa.get("evidence")),
                    meta={"category": "" if category is None else str(category)},
                )
            )
    return contexts, questions
