"""`oracle_gold` is the ceiling and must not leak; `long_context` sees everything (§2.11)."""

from pathlib import Path
from typing import Any

import pytest

from orchestrator.artifacts import JsonlArtifact
from reference.baselines import inject_retrieval

UID = "eval:r1:d:c0"


def chunk(chunk_id: str, content: str, session_id: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "user", "content": content}
    if session_id:
        message["session_id"] = session_id
    return {
        "kind": "chunk",
        "id": f"{UID}:chunk-{chunk_id}",
        "user_id": UID,
        "ctx_id": "c0",
        "chunk_id": chunk_id,
        "messages": [message],
    }


def question(evidence: list[str], handles: list[str] | None = None) -> dict[str, Any]:
    """A question record as `plan` writes it: dataset handles plus resolved chunk ids."""
    return {
        "kind": "question",
        "id": f"{UID}:q-q1",
        "user_id": UID,
        "ctx_id": "c0",
        "q_id": "q1",
        "question": "where did the cat sit?",
        "gold": ["on the mat"],
        "evidence_chunk_ids": handles if handles is not None else evidence,
        "evidence_chunks": evidence,
        "meta": {},
    }


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    plan = JsonlArtifact(tmp_path / "plan.jsonl")
    plan.append(chunk("c0", "the cat sat on the mat"))
    plan.append(chunk("c1", "the dog barked at the postman"))
    plan.append(chunk("c2", "unrelated small talk"))
    plan.append(question(["c0"]))
    return tmp_path


def rows(run_dir: Path) -> list[dict[str, Any]]:
    return list(JsonlArtifact(run_dir / "retrieve.jsonl").stream())


def test_oracle_gold_retrieves_exactly_the_gold_evidence_and_nothing_else(run_dir: Path) -> None:
    """A leaky ceiling silently flatters every system beneath it."""
    inject_retrieval(run_dir, "oracle_gold", top_k=10)
    memories = rows(run_dir)[0]["memories"]
    assert [m["source_ids"] for m in memories] == [["c0"]]


def test_oracle_gold_content_is_the_gold_chunk(run_dir: Path) -> None:
    inject_retrieval(run_dir, "oracle_gold", top_k=10)
    assert rows(run_dir)[0]["memories"][0]["content"] == "the cat sat on the mat"


def test_long_context_retrieves_the_whole_corpus(run_dir: Path) -> None:
    inject_retrieval(run_dir, "long_context", top_k=10)
    memories = rows(run_dir)[0]["memories"]
    assert [m["source_ids"] for m in memories] == [["c0"], ["c1"], ["c2"]]


def test_both_baselines_honour_top_k(run_dir: Path) -> None:
    inject_retrieval(run_dir, "long_context", top_k=2)
    assert len(rows(run_dir)[0]["memories"]) == 2


def test_records_are_shaped_like_a_real_retrieval(run_dir: Path) -> None:
    """Downstream stages must not be able to tell the difference."""
    inject_retrieval(run_dir, "oracle_gold", top_k=10)
    record = rows(run_dir)[0]
    assert set(record) >= {"id", "user_id", "query", "top_k", "attempts", "latency_ms", "memories"}
    assert record["query"] == "where did the cat sit?"


def test_the_row_records_which_baseline_produced_it(run_dir: Path) -> None:
    inject_retrieval(run_dir, "oracle_gold", top_k=10)
    assert rows(run_dir)[0]["in_process"] == "oracle_gold"


def test_session_level_evidence_expands_to_its_chunks(tmp_path: Path) -> None:
    """LongMemEval names sessions; `plan` resolved sess0 onto both of its chunks."""
    plan = JsonlArtifact(tmp_path / "plan.jsonl")
    plan.append(chunk("c0", "part one of the session", session_id="sess0"))
    plan.append(chunk("c1", "part two of the session", session_id="sess0"))
    plan.append(chunk("c2", "a different session", session_id="sess1"))
    plan.append(question(["c0", "c1"], handles=["sess0"]))

    inject_retrieval(tmp_path, "oracle_gold", top_k=10)
    assert [m["source_ids"] for m in rows(tmp_path)[0]["memories"]] == [["c0"], ["c1"]]


def test_a_question_with_no_evidence_gets_an_empty_oracle_retrieval(tmp_path: Path) -> None:
    plan = JsonlArtifact(tmp_path / "plan.jsonl")
    plan.append(chunk("c0", "content"))
    plan.append(question([]))
    inject_retrieval(tmp_path, "oracle_gold", top_k=10)
    assert rows(tmp_path)[0]["memories"] == []


def test_injection_is_resumable(run_dir: Path) -> None:
    inject_retrieval(run_dir, "oracle_gold", top_k=10)
    before = (run_dir / "retrieve.jsonl").read_bytes()
    assert inject_retrieval(run_dir, "oracle_gold", top_k=10) == 0
    assert (run_dir / "retrieve.jsonl").read_bytes() == before


def test_only_the_questions_own_context_is_visible(tmp_path: Path) -> None:
    """long_context sees the whole corpus of its own context, not another user's."""
    plan = JsonlArtifact(tmp_path / "plan.jsonl")
    plan.append(chunk("c0", "mine"))
    plan.append(
        {
            "kind": "chunk",
            "id": "eval:r1:d:other:chunk-c0",
            "user_id": "eval:r1:d:other",
            "ctx_id": "other",
            "chunk_id": "c0",
            "messages": [{"role": "user", "content": "someone else's haystack"}],
        }
    )
    plan.append(question([]))
    inject_retrieval(tmp_path, "long_context", top_k=10)
    contents = [m["content"] for m in rows(tmp_path)[0]["memories"]]
    assert contents == ["mine"]
