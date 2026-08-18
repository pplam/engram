"""`judge` — label each generated answer with the resolved judge model (ARCH §5, D7).

An unparseable judge reply is recorded as UNPARSEABLE, never silently as WRONG: a
judge failure and a wrong answer are different findings.
"""

import json
import re
import time
from pathlib import Path

from orchestrator.artifacts import JsonlArtifact
from orchestrator.models import ChatClient
from orchestrator.plan_records import QuestionRecord, parse_plan_record
from orchestrator.prompting import render
from orchestrator.runlog import get_logger

log = get_logger("judge")

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_label(reply: str) -> tuple[str, bool]:
    """Return the (label, is_correct) pair for a judge reply."""
    text = reply.strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    found = _OBJECT.search(text)
    if found:
        try:
            payload = json.loads(found.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            label = str(payload.get("label", "")).upper()
            if label in ("CORRECT", "WRONG"):
                return label, label == "CORRECT"
    return "UNPARSEABLE", False


def _questions(run_dir: Path) -> dict[str, QuestionRecord]:
    plan = JsonlArtifact(run_dir / "plan.jsonl")
    records = (parse_plan_record(raw) for raw in plan.stream())
    return {r.id: r for r in records if isinstance(r, QuestionRecord)}


async def run_judge(
    run_dir: Path,
    chat: ChatClient,
    model: str,
    temperature: float,
    prompt: str,
) -> int:
    """Judge every not-yet-judged answer; return how many were written."""
    answers = JsonlArtifact(run_dir / "answer.jsonl")
    out = JsonlArtifact(run_dir / "judge.jsonl")
    done = out.read_ids()
    questions = _questions(run_dir)
    written = 0
    if done:
        log.info("resuming: %d judgements already present", len(done))

    for row in answers.stream():
        record_id = str(row["id"])
        if record_id in done:
            continue
        question = questions[record_id]
        rendered = render(
            prompt,
            question=question.question,
            gold=" | ".join(question.gold),
            prediction=str(row.get("generated_answer", "")),
        )
        started = time.monotonic()
        completion = await chat.chat(
            [{"role": "user", "content": rendered}],
            model=model,
            temperature=temperature,
        )
        label, is_correct = parse_label(completion.text)
        # The label is a verdict, so it belongs here; the judge's prose does not. An
        # UNPARSEABLE reply is a judge failure, and this is where it becomes visible.
        log.info(
            "judge id=%s model=%s label=%s latency_ms=%d prompt_tokens=%d completion_tokens=%d",
            record_id,
            model,
            label,
            int((time.monotonic() - started) * 1000),
            completion.prompt_tokens,
            completion.completion_tokens,
        )
        out.append(
            {
                "id": record_id,
                "user_id": question.user_id,
                "label": label,
                "is_correct": is_correct,
                "judge_response": completion.text,
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
            }
        )
        written += 1

    return written
