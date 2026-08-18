"""`ingest` and `retrieve`: thin, resumable, and verbatim-preserving (§2.7).

The system under test is a fake adapter, so nothing here needs a socket. What the
stages record is what `score` reads offline, so the record shape is part of the
contract between them and is pinned here.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from contract.adapter import ContractViolation, Memory, Message
from orchestrator.artifacts import JsonlArtifact
from orchestrator.client import AdapterClient, RetryPolicy
from orchestrator.stages.ingest import run_ingest
from orchestrator.stages.retrieve import run_retrieve
from reference.fakes import FakeMemory

POLICY = RetryPolicy(attempts=2, timeout_s=5, workers=4, backoff_s=0.0)

UID = "eval:r1:d:c0"

PLAN: list[dict[str, Any]] = [
    {
        "kind": "chunk",
        "id": f"{UID}:chunk-c0",
        "user_id": UID,
        "ctx_id": "c0",
        "chunk_id": "c0",
        "messages": [{"role": "user", "content": "the cat sat on the mat", "timestamp": 1}],
    },
    {
        "kind": "chunk",
        "id": f"{UID}:chunk-c1",
        "user_id": UID,
        "ctx_id": "c0",
        "chunk_id": "c1",
        "messages": [{"role": "user", "content": "the dog barked loudly", "timestamp": 2}],
    },
    {
        "kind": "question",
        "id": f"{UID}:q-q1",
        "user_id": UID,
        "ctx_id": "c0",
        "q_id": "q1",
        "question": "where did the cat sit?",
        "gold": ["on the mat"],
        "evidence_chunk_ids": ["c0"],
        "meta": {},
    },
]


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    artifact = JsonlArtifact(tmp_path / "plan.jsonl")
    for record in PLAN:
        artifact.append(record)
    return tmp_path


def client(adapter: object | None = None, **kw: object) -> AdapterClient:
    policy = POLICY.model_copy(update=kw) if kw else POLICY
    return AdapterClient(adapter or FakeMemory(), policy=policy)  # type: ignore[arg-type]


def records(run_dir: Path, name: str) -> list[dict[str, Any]]:
    return list(JsonlArtifact(run_dir / name).stream())


async def test_ingest_writes_one_record_per_chunk(run_dir: Path) -> None:
    await run_ingest(run_dir, client())
    assert len(records(run_dir, "ingest.jsonl")) == 2


async def test_ingest_ignores_question_records(run_dir: Path) -> None:
    await run_ingest(run_dir, client())
    assert all(":chunk-" in str(r["id"]) for r in records(run_dir, "ingest.jsonl"))


async def test_ingest_records_status_attempts_and_latency(run_dir: Path) -> None:
    await run_ingest(run_dir, client())
    record = records(run_dir, "ingest.jsonl")[0]
    assert record["status"] == "ok"
    assert record["attempts"] == 1
    assert isinstance(record["latency_ms"], int)


@pytest.mark.parametrize("field", ["http_status", "bytes"])
async def test_ingest_records_no_http_specific_fields(run_dir: Path, field: str) -> None:
    """There is no wire any more; a status code would be a fiction for a local adapter."""
    await run_ingest(run_dir, client())
    assert field not in records(run_dir, "ingest.jsonl")[0]


async def test_ingest_passes_messages_as_typed_records(run_dir: Path) -> None:
    seen: list[Sequence[Message]] = []

    class Recording(FakeMemory):
        async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> None:
            seen.append(messages)
            await super().add(user_id, messages, chunk_id)

    await run_ingest(run_dir, client(Recording()))
    assert all(isinstance(m, Message) for batch in seen for m in batch)
    assert seen[0][0].content == "the cat sat on the mat"
    assert seen[0][0].timestamp == 1


async def test_ingest_passes_the_chunk_id_as_provenance(run_dir: Path) -> None:
    seen: list[str] = []

    class Recording(FakeMemory):
        async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> None:
            seen.append(chunk_id)
            await super().add(user_id, messages, chunk_id)

    await run_ingest(run_dir, client(Recording()))
    assert sorted(seen) == ["c0", "c1"]


async def test_ingest_is_resumable_and_adds_nothing_on_a_second_run(run_dir: Path) -> None:
    adapter = FakeMemory()
    await run_ingest(run_dir, client(adapter))
    before = (run_dir / "ingest.jsonl").read_bytes()
    await run_ingest(run_dir, client(adapter))
    assert (run_dir / "ingest.jsonl").read_bytes() == before


async def test_ingest_resumes_only_the_missing_chunks(run_dir: Path) -> None:
    JsonlArtifact(run_dir / "ingest.jsonl").append(
        {"id": f"{UID}:chunk-c0", "user_id": UID, "status": "ok"}
    )
    await run_ingest(run_dir, client())
    ids = [r["id"] for r in records(run_dir, "ingest.jsonl")]
    assert ids == [f"{UID}:chunk-c0", f"{UID}:chunk-c1"]


async def test_ingest_fails_loudly_on_a_contract_violation(run_dir: Path) -> None:
    class Broken(FakeMemory):
        async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> None:
            raise ContractViolation("cannot store")

    with pytest.raises(ContractViolation):
        await run_ingest(run_dir, client(Broken()))


async def test_a_violating_ingest_does_not_record_a_success(run_dir: Path) -> None:
    class Broken(FakeMemory):
        async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> None:
            raise ContractViolation("cannot store")

    with pytest.raises(ContractViolation):
        await run_ingest(run_dir, client(Broken()))
    assert not any(r.get("status") == "ok" for r in records(run_dir, "ingest.jsonl"))


async def test_retrieve_writes_one_record_per_question(run_dir: Path) -> None:
    adapter = FakeMemory()
    await run_ingest(run_dir, client(adapter))
    await run_retrieve(run_dir, client(adapter), top_k=10)
    assert len(records(run_dir, "retrieve.jsonl")) == 1


async def test_retrieve_records_the_query_and_budget_it_used(run_dir: Path) -> None:
    adapter = FakeMemory()
    await run_ingest(run_dir, client(adapter))
    await run_retrieve(run_dir, client(adapter), top_k=7)
    record = records(run_dir, "retrieve.jsonl")[0]
    assert record["query"] == "where did the cat sit?"
    assert record["top_k"] == 7


async def test_retrieve_stores_memories_under_the_memories_key(run_dir: Path) -> None:
    adapter = FakeMemory()
    await run_ingest(run_dir, client(adapter))
    await run_retrieve(run_dir, client(adapter), top_k=10)
    record = records(run_dir, "retrieve.jsonl")[0]
    assert record["memories"][0]["source_ids"] == ["c0"]


async def test_a_retrieve_record_carries_only_the_three_memory_fields(run_dir: Path) -> None:
    """The record is what a third party rescores from, so its shape is pinned."""
    adapter = FakeMemory()
    await run_ingest(run_dir, client(adapter))
    await run_retrieve(run_dir, client(adapter), top_k=10)
    memory = records(run_dir, "retrieve.jsonl")[0]["memories"][0]
    assert set(memory) == {"content", "score", "source_ids"}


async def test_a_retrieve_record_has_no_duplicate_response_blob(run_dir: Path) -> None:
    """`data` and `response` used to hold the same list twice; nothing read the second."""
    adapter = FakeMemory()
    await run_ingest(run_dir, client(adapter))
    await run_retrieve(run_dir, client(adapter), top_k=10)
    record = records(run_dir, "retrieve.jsonl")[0]
    assert "response" not in record
    assert "data" not in record


async def test_retrieve_preserves_the_rank_order_the_system_returned(run_dir: Path) -> None:
    class Ranked(FakeMemory):
        async def search(self, user_id: str, query: str, top_k: int) -> Sequence[Memory]:
            return [
                Memory(content="first", score=0.1, source_ids=("c1",)),
                Memory(content="second", score=0.9, source_ids=("c0",)),
            ]

    await run_retrieve(run_dir, client(Ranked()), top_k=10)
    memories = records(run_dir, "retrieve.jsonl")[0]["memories"]
    assert [m["content"] for m in memories] == ["first", "second"]


async def test_retrieve_is_resumable(run_dir: Path) -> None:
    adapter = FakeMemory()
    await run_ingest(run_dir, client(adapter))
    await run_retrieve(run_dir, client(adapter), top_k=10)
    before = (run_dir / "retrieve.jsonl").read_bytes()
    await run_retrieve(run_dir, client(adapter), top_k=10)
    assert (run_dir / "retrieve.jsonl").read_bytes() == before


async def test_retrieve_records_latency_and_attempts(run_dir: Path) -> None:
    adapter = FakeMemory()
    await run_ingest(run_dir, client(adapter))
    await run_retrieve(run_dir, client(adapter), top_k=10)
    record = records(run_dir, "retrieve.jsonl")[0]
    assert record["attempts"] == 1
    assert isinstance(record["latency_ms"], int)


async def test_retrieve_fails_loudly_when_the_budget_is_exceeded(run_dir: Path) -> None:
    adapter = FakeMemory(overshoot_top_k=True)
    await run_ingest(run_dir, client(adapter))
    with pytest.raises(ContractViolation, match="top_k"):
        await run_retrieve(run_dir, client(adapter), top_k=1)


async def test_retrieve_does_not_rewrite_the_query(run_dir: Path) -> None:
    """The query is the verbatim task question; the harness never rephrases it."""
    seen: list[str] = []

    class Recording(FakeMemory):
        async def search(self, user_id: str, query: str, top_k: int) -> Sequence[Memory]:
            seen.append(query)
            return ()

    await run_retrieve(run_dir, client(Recording()), top_k=10)
    assert seen == ["where did the cat sit?"]


async def test_retrieve_searches_under_the_questions_user_id(run_dir: Path) -> None:
    """The isolation boundary: a search must be scoped to the question's own namespace."""
    seen: list[str] = []

    class Recording(FakeMemory):
        async def search(self, user_id: str, query: str, top_k: int) -> Sequence[Memory]:
            seen.append(user_id)
            return ()

    await run_retrieve(run_dir, client(Recording()), top_k=10)
    assert seen == [UID]


async def test_a_question_with_no_memories_still_gets_a_record(run_dir: Path) -> None:
    """An empty retrieval is a measurement, not a missing row — `no_memory` is all of them."""
    await run_retrieve(run_dir, client(FakeMemory()), top_k=10)
    assert records(run_dir, "retrieve.jsonl")[0]["memories"] == []


async def test_ingest_on_a_plan_with_no_chunks_writes_nothing(tmp_path: Path) -> None:
    JsonlArtifact(tmp_path / "plan.jsonl").append(
        {
            "kind": "question",
            "id": f"{UID}:q-q1",
            "user_id": UID,
            "ctx_id": "c0",
            "q_id": "q1",
            "question": "q?",
            "gold": ["a"],
            "evidence_chunk_ids": [],
            "meta": {},
        }
    )
    await run_ingest(tmp_path, client())
    assert not (tmp_path / "ingest.jsonl").exists()
