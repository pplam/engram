"""`answer` and `judge`: pinned prompts, temperature 0, resumable (§2.9)."""

from pathlib import Path
from typing import Any

import pytest

from orchestrator.artifacts import JsonlArtifact
from orchestrator.models import StubChat
from orchestrator.stages.answer import run_answer
from orchestrator.stages.judge import run_judge

UID = "eval:r1:d:c0"
QID = f"{UID}:q-q1"

PLAN: list[dict[str, Any]] = [
    {
        "kind": "question",
        "id": QID,
        "user_id": UID,
        "ctx_id": "c0",
        "q_id": "q1",
        "question": "where did the cat sit?",
        "gold": ["on the mat"],
        "evidence_chunk_ids": ["c0"],
        "meta": {},
    }
]

RETRIEVE: list[dict[str, Any]] = [
    {
        "id": QID,
        "user_id": UID,
        "query": "where did the cat sit?",
        "top_k": 10,
        "attempts": 1,
        "latency_ms": 5,
        "data": [{"id": "mem_1", "content": "the cat sat on the mat", "score": 0.9}],
        "response": {"data": [{"id": "mem_1", "content": "the cat sat on the mat"}]},
    }
]


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    for name, rows in (("plan.jsonl", PLAN), ("retrieve.jsonl", RETRIEVE)):
        artifact = JsonlArtifact(tmp_path / name)
        for row in rows:
            artifact.append(row)
    return tmp_path


def records(run_dir: Path, name: str) -> list[dict[str, Any]]:
    return list(JsonlArtifact(run_dir / name).stream())


ANSWER_PROMPT = "Memories:\n{memories}\n\nQuestion: {question}\n\nAnswer:"
JUDGE_PROMPT = 'Q: {question}\nGold: {gold}\nPred: {prediction}\nReply {{"label": ...}}'


async def test_answer_writes_one_record_per_retrieval(run_dir: Path) -> None:
    chat = StubChat(replies=["on the mat"])
    await run_answer(run_dir, chat, model="m", temperature=0, prompt=ANSWER_PROMPT, seed=7)
    assert len(records(run_dir, "answer.jsonl")) == 1


async def test_answer_records_the_generated_text_and_token_counts(run_dir: Path) -> None:
    chat = StubChat(replies=["on the mat"], prompt_tokens=12, completion_tokens=4)
    await run_answer(run_dir, chat, model="m", temperature=0, prompt=ANSWER_PROMPT, seed=7)
    record = records(run_dir, "answer.jsonl")[0]
    assert record["generated_answer"] == "on the mat"
    assert record["prompt_tokens"] == 12
    assert record["completion_tokens"] == 4


async def test_answer_uses_the_pinned_model_temperature_and_seed(run_dir: Path) -> None:
    chat = StubChat(replies=["x"])
    await run_answer(run_dir, chat, model="pinned", temperature=0, prompt=ANSWER_PROMPT, seed=7)
    assert chat.calls[0].model == "pinned"
    assert chat.calls[0].temperature == 0
    assert chat.calls[0].seed == 7


async def test_answer_prompt_carries_the_retrieved_content(run_dir: Path) -> None:
    chat = StubChat(replies=["x"])
    await run_answer(run_dir, chat, model="m", temperature=0, prompt=ANSWER_PROMPT)
    sent = chat.calls[0].messages[-1]["content"]
    assert "the cat sat on the mat" in sent
    assert "where did the cat sit?" in sent


async def test_answer_truncates_memories_to_the_retrieved_order(run_dir: Path) -> None:
    """Rank order from the system is preserved in the prompt."""
    JsonlArtifact(run_dir / "retrieve.jsonl").path.write_text("")
    artifact = JsonlArtifact(run_dir / "retrieve.jsonl")
    artifact.append(
        {
            **RETRIEVE[0],
            "data": [
                {"id": "m1", "content": "first memory"},
                {"id": "m2", "content": "second memory"},
            ],
        }
    )
    chat = StubChat(replies=["x"])
    await run_answer(run_dir, chat, model="m", temperature=0, prompt=ANSWER_PROMPT)
    sent = chat.calls[0].messages[-1]["content"]
    assert sent.index("first memory") < sent.index("second memory")


async def test_answer_handles_an_empty_retrieval(run_dir: Path) -> None:
    artifact = JsonlArtifact(run_dir / "retrieve.jsonl")
    artifact.path.write_text("")
    artifact.append({**RETRIEVE[0], "data": []})
    chat = StubChat(replies=["I don't know."])
    await run_answer(run_dir, chat, model="m", temperature=0, prompt=ANSWER_PROMPT)
    assert records(run_dir, "answer.jsonl")[0]["generated_answer"] == "I don't know."


async def test_answer_is_resumable(run_dir: Path) -> None:
    chat = StubChat(replies=["a"])
    await run_answer(run_dir, chat, model="m", temperature=0, prompt=ANSWER_PROMPT)
    before = (run_dir / "answer.jsonl").read_bytes()
    await run_answer(run_dir, chat, model="m", temperature=0, prompt=ANSWER_PROMPT)
    assert (run_dir / "answer.jsonl").read_bytes() == before
    assert len(chat.calls) == 1


async def test_judge_writes_a_binary_label(run_dir: Path) -> None:
    await seed_answer(run_dir, "on the mat")
    chat = StubChat(replies=['{"label": "CORRECT", "reason": "same place"}'])
    await run_judge(run_dir, chat, model="j", temperature=0, prompt=JUDGE_PROMPT)
    record = records(run_dir, "judge.jsonl")[0]
    assert record["label"] == "CORRECT"
    assert record["is_correct"] is True


async def test_judge_records_a_wrong_label(run_dir: Path) -> None:
    await seed_answer(run_dir, "in the garden")
    chat = StubChat(replies=['{"label": "WRONG", "reason": "different place"}'])
    await run_judge(run_dir, chat, model="j", temperature=0, prompt=JUDGE_PROMPT)
    record = records(run_dir, "judge.jsonl")[0]
    assert record["label"] == "WRONG"
    assert record["is_correct"] is False


async def test_judge_stores_the_raw_judge_response(run_dir: Path) -> None:
    await seed_answer(run_dir, "on the mat")
    raw = '{"label": "CORRECT", "reason": "same place"}'
    chat = StubChat(replies=[raw])
    await run_judge(run_dir, chat, model="j", temperature=0, prompt=JUDGE_PROMPT)
    assert records(run_dir, "judge.jsonl")[0]["judge_response"] == raw


async def test_judge_prompt_carries_question_gold_and_prediction(run_dir: Path) -> None:
    await seed_answer(run_dir, "on the mat")
    chat = StubChat(replies=['{"label": "CORRECT"}'])
    await run_judge(run_dir, chat, model="j", temperature=0, prompt=JUDGE_PROMPT)
    sent = chat.calls[0].messages[-1]["content"]
    assert "where did the cat sit?" in sent
    assert "on the mat" in sent


async def test_judge_uses_the_resolved_judge_model(run_dir: Path) -> None:
    await seed_answer(run_dir, "on the mat")
    chat = StubChat(replies=['{"label": "CORRECT"}'])
    await run_judge(run_dir, chat, model="qwen3-14b", temperature=0, prompt=JUDGE_PROMPT)
    assert chat.calls[0].model == "qwen3-14b"


async def test_judge_tolerates_a_fenced_json_reply(run_dir: Path) -> None:
    await seed_answer(run_dir, "on the mat")
    chat = StubChat(replies=['```json\n{"label": "CORRECT"}\n```'])
    await run_judge(run_dir, chat, model="j", temperature=0, prompt=JUDGE_PROMPT)
    assert records(run_dir, "judge.jsonl")[0]["is_correct"] is True


async def test_an_unparseable_judge_reply_is_recorded_as_unparseable_not_wrong(
    run_dir: Path,
) -> None:
    """A judge failure must be distinguishable from a wrong answer."""
    await seed_answer(run_dir, "on the mat")
    chat = StubChat(replies=["I cannot comply"])
    await run_judge(run_dir, chat, model="j", temperature=0, prompt=JUDGE_PROMPT)
    record = records(run_dir, "judge.jsonl")[0]
    assert record["label"] == "UNPARSEABLE"
    assert record["is_correct"] is False


async def test_judge_is_resumable(run_dir: Path) -> None:
    await seed_answer(run_dir, "on the mat")
    chat = StubChat(replies=['{"label": "CORRECT"}'])
    await run_judge(run_dir, chat, model="j", temperature=0, prompt=JUDGE_PROMPT)
    before = (run_dir / "judge.jsonl").read_bytes()
    await run_judge(run_dir, chat, model="j", temperature=0, prompt=JUDGE_PROMPT)
    assert (run_dir / "judge.jsonl").read_bytes() == before
    assert len(chat.calls) == 1


async def test_judge_compares_against_every_acceptable_gold_answer(run_dir: Path) -> None:
    artifact = JsonlArtifact(run_dir / "plan.jsonl")
    artifact.path.write_text("")
    artifact.append({**PLAN[0], "gold": ["on the mat", "the mat"]})
    await seed_answer(run_dir, "the mat")
    chat = StubChat(replies=['{"label": "CORRECT"}'])
    await run_judge(run_dir, chat, model="j", temperature=0, prompt=JUDGE_PROMPT)
    sent = chat.calls[0].messages[-1]["content"]
    assert "on the mat" in sent
    assert "the mat" in sent


async def seed_answer(run_dir: Path, text: str) -> None:
    JsonlArtifact(run_dir / "answer.jsonl").append(
        {
            "id": QID,
            "user_id": UID,
            "generated_answer": text,
            "prompt_tokens": 10,
            "completion_tokens": 3,
        }
    )
