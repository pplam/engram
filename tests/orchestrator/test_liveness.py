"""The post-ingest gate: a system that stored nothing must fail before more is spent.

An `add` that answers 200 having stored nothing is the failure this catches. It is not
hypothetical — mem0 returns an empty extraction, and therefore HTTP 200, whenever its
completion budget is too small for the pinned model to emit any content. Twenty of
twenty-one chunks landed nowhere and the run still scored, reporting the absence as a
retrieval miss rather than as the broken integration it was.

The probe is a real planned question rather than a synthetic string: a store filtering
by score threshold can legitimately return nothing for a nonsense query, and a gate that
fires on a healthy system is worse than no gate.
"""

from pathlib import Path
from typing import Any

import pytest

from contract.adapter import ContractViolation
from orchestrator.artifacts import JsonlArtifact
from orchestrator.client import AdapterClient, RetryPolicy
from orchestrator.stages.ingest import run_ingest
from orchestrator.stages.liveness import run_liveness
from reference.baselines.adapters import NoMemoryAdapter
from reference.fakes import FakeMemory

POLICY = RetryPolicy(attempts=2, timeout_s=5, workers=4, backoff_s=0.0)
UID = "eval:r1:d:c0"


def plan_records(user_id: str = UID) -> list[dict[str, Any]]:
    return [
        {
            "kind": "chunk",
            "id": f"{user_id}:chunk-c0",
            "user_id": user_id,
            "ctx_id": "c0",
            "chunk_id": "c0",
            # Fact-shaped: the extractive fake keeps only what asserts something about
            # the user, and a probe it discards would test the fixture, not the gate.
            "messages": [{"role": "user", "content": "my cat is called Mochi"}],
        },
        {
            "kind": "question",
            "id": f"{user_id}:q-q1",
            "user_id": user_id,
            "ctx_id": "c0",
            "q_id": "q1",
            "question": "what is the cat called?",
            "gold": ["Mochi"],
            "evidence_chunk_ids": ["c0"],
            "evidence_chunks": ["c0"],
            "meta": {},
        },
    ]


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    artifact = JsonlArtifact(tmp_path / "plan.jsonl")
    for record in plan_records():
        artifact.append(record)
    return tmp_path


def client(adapter: object) -> AdapterClient:
    return AdapterClient(adapter, policy=POLICY)  # type: ignore[arg-type]


async def test_a_store_that_kept_nothing_fails_the_run(run_dir: Path) -> None:
    adapter = FakeMemory(store_nothing=True)
    await run_ingest(run_dir, client(adapter))
    with pytest.raises(ContractViolation, match="stored nothing"):
        await run_liveness(run_dir, client(adapter), top_k=10)


async def test_the_failure_names_what_was_written(run_dir: Path) -> None:
    """A bare "empty" would send someone reading it to the wrong layer."""
    adapter = FakeMemory(store_nothing=True)
    await run_ingest(run_dir, client(adapter))
    with pytest.raises(ContractViolation, match="1 chunk"):
        await run_liveness(run_dir, client(adapter), top_k=10)


async def test_a_healthy_store_passes(run_dir: Path) -> None:
    adapter = FakeMemory()
    await run_ingest(run_dir, client(adapter))
    assert await run_liveness(run_dir, client(adapter), top_k=10) == 1


async def test_an_extractive_store_passes(run_dir: Path) -> None:
    """Keeping only what reads as a fact is a second conformant shape, not a failure."""
    adapter = FakeMemory(extractive=True)
    await run_ingest(run_dir, client(adapter))
    assert await run_liveness(run_dir, client(adapter), top_k=10) == 1


async def test_a_store_reporting_no_source_ids_still_passes(run_dir: Path) -> None:
    """Provenance is opt-in (ARCH §7.2); the gate is presence, not provenance."""
    adapter = FakeMemory(drop_source_ids=True)
    await run_ingest(run_dir, client(adapter))
    assert await run_liveness(run_dir, client(adapter), top_k=10) == 1


async def test_a_user_with_no_questions_is_not_probed(tmp_path: Path) -> None:
    """The probe has to be a real question, so a context without one cannot be checked."""
    artifact = JsonlArtifact(tmp_path / "plan.jsonl")
    artifact.append(plan_records()[0])
    adapter = FakeMemory(store_nothing=True)
    await run_ingest(tmp_path, client(adapter))
    assert await run_liveness(tmp_path, client(adapter), top_k=10) == 0


async def test_a_resume_that_wrote_nothing_is_not_probed(run_dir: Path) -> None:
    """Ingest skips chunks already recorded, so a completed resume writes nothing.

    The gate vouches for what this invocation wrote. With nothing written there is
    nothing to vouch for, and the store's contents were settled by the earlier run.
    """
    adapter = FakeMemory(store_nothing=True)
    await run_ingest(run_dir, client(adapter))
    assert await run_liveness(run_dir, client(adapter), top_k=10, ingested=0) == 0


async def test_a_system_declaring_it_stores_nothing_is_not_failed(run_dir: Path) -> None:
    """`no_memory` is the floor baseline: storing nothing is its measurement, not a fault.

    Declared by the adapter rather than decided by name in the orchestrator, which may
    carry no per-system branching (CLAUDE.md).
    """
    adapter = NoMemoryAdapter()
    await run_ingest(run_dir, client(adapter))
    assert await run_liveness(run_dir, client(adapter), top_k=10, stores_nothing=True) == 0


async def test_the_declaration_is_read_off_the_adapter(run_dir: Path) -> None:
    """The floor baseline carries the marker; a system that merely fails must not."""
    assert getattr(NoMemoryAdapter(), "stores_nothing", False) is True
    assert getattr(FakeMemory(store_nothing=True), "stores_nothing", False) is False


async def test_the_probe_is_bounded_rather_than_one_search_per_context(
    tmp_path: Path,
) -> None:
    """LongMemEval-S has one context per question, so probing all of them is the cost
    of a second retrieve pass. A misconfiguration is global, so a sample catches it."""
    artifact = JsonlArtifact(tmp_path / "plan.jsonl")
    for n in range(12):
        for record in plan_records(user_id=f"eval:r1:d:c{n}"):
            artifact.append(record)
    adapter = FakeMemory()
    await run_ingest(tmp_path, client(adapter))
    assert await run_liveness(tmp_path, client(adapter), top_k=10, sample=5) == 5
