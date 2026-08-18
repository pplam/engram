"""`score` is pure and offline; report.json is the publication unit (§2.10, §7)."""

import json
from pathlib import Path
from typing import Any

import pytest

from orchestrator.artifacts import JsonlArtifact, read_json, write_json
from orchestrator.report import Report
from orchestrator.stages.score import run_score

UID = "eval:r1:d:c0"


def qid(n: int) -> str:
    return f"{UID}:q-q{n}"


MANIFEST: dict[str, Any] = {
    "run_id": "r1",
    "system": {"name": "mock"},
    "suite": {"version": "v1", "sha256": "abc"},
    "dataset": {
        "name": "fixture_many",
        "version": "1.0",
        "source_sha256": "def",
        "scorer": "judge_binary",
        "task_type": "open_qa",
        "evidence_granularity": "message",
        "supports_recall": True,
    },
    "models": {
        "chat": "gpt-4o-mini",
        "embedding": "text-embedding-3-small",
        "judge": "gpt-4.1-mini",
        "judge_source": "suite",
    },
    "prompts": {"answer_sha256": "a1", "answer_source": "suite", "judge_sha256": "j1"},
    "top_k": 10,
    "limit": None,
}


def build(tmp_path: Path, **overrides: Any) -> Path:
    """Write a small but complete artifact set for two questions."""
    write_json(tmp_path / "manifest.json", {**MANIFEST, **overrides.pop("manifest", {})})

    plan = JsonlArtifact(tmp_path / "plan.jsonl")
    plan.append(
        {
            "kind": "chunk",
            "id": f"{UID}:chunk-c0",
            "user_id": UID,
            "ctx_id": "c0",
            "chunk_id": "c0",
            "messages": [{"role": "user", "content": "the cat sat on the mat"}],
        }
    )
    for n, gold in ((1, ["on the mat"]), (2, ["Mochi"])):
        plan.append(
            {
                "kind": "question",
                "id": qid(n),
                "user_id": UID,
                "ctx_id": "c0",
                "q_id": f"q{n}",
                "question": f"question {n}?",
                "gold": gold,
                "evidence_chunk_ids": ["m0"],
                "evidence_chunks": ["c0"],
                "meta": {},
            }
        )

    ingest = JsonlArtifact(tmp_path / "ingest.jsonl")
    ingest.append(
        {
            "id": f"{UID}:chunk-c0",
            "user_id": UID,
            "chunk_id": "c0",
            "status": "ok",
            "attempts": 2,
            "latency_ms": 100,
            "started_at_ms": 1_000,
            "ended_at_ms": 1_100,
            "bytes": 64,
            "http_status": 200,
        }
    )

    retrieve = JsonlArtifact(tmp_path / "retrieve.jsonl")
    for n, sources in ((1, ["c0"]), (2, ["c9"])):
        retrieve.append(
            {
                "id": qid(n),
                "user_id": UID,
                "query": f"question {n}?",
                "top_k": 10,
                "attempts": 1,
                "latency_ms": 10 * n,
                "data": [
                    {"id": "m1", "content": "the cat sat on the mat", "x_source_ids": sources}
                ],
                "response": {"data": []},
            }
        )

    answer = JsonlArtifact(tmp_path / "answer.jsonl")
    for n in (1, 2):
        answer.append(
            {
                "id": qid(n),
                "user_id": UID,
                "generated_answer": f"answer {n}",
                "prompt_tokens": 100,
                "completion_tokens": 10,
            }
        )

    judge = JsonlArtifact(tmp_path / "judge.jsonl")
    for n, label in ((1, "CORRECT"), (2, "WRONG")):
        judge.append(
            {
                "id": qid(n),
                "user_id": UID,
                "label": label,
                "is_correct": label == "CORRECT",
                "judge_response": "{}",
                "prompt_tokens": 50,
                "completion_tokens": 5,
            }
        )
    return tmp_path


def test_writes_report_json(tmp_path: Path) -> None:
    run_score(build(tmp_path))
    assert (tmp_path / "report.json").is_file()


def test_report_validates_against_its_model(tmp_path: Path) -> None:
    run_score(build(tmp_path))
    Report.model_validate(read_json(tmp_path / "report.json"))


def test_quality_accuracy_is_the_fraction_judged_correct(tmp_path: Path) -> None:
    report = run_score(build(tmp_path))
    assert report.row.quality.accuracy == 0.5
    assert report.row.quality.questions == 2
    assert report.row.quality.correct == 1


def test_row_names_the_system_dataset_and_suite(tmp_path: Path) -> None:
    report = run_score(build(tmp_path))
    assert report.row.system == "mock"
    assert report.row.dataset == "fixture_many"
    assert report.row.suite == "v1"


def test_row_carries_the_resolved_judge_and_its_source(tmp_path: Path) -> None:
    """A reader must never have to infer which judge produced a number."""
    report = run_score(build(tmp_path))
    assert report.row.quality.judge == "gpt-4.1-mini"
    assert report.row.quality.judge_source == "suite"


def test_row_carries_a_dataset_pinned_judge(tmp_path: Path) -> None:
    manifest = {**MANIFEST["models"], "judge": "qwen3-14b", "judge_source": "dataset"}
    report = run_score(build(tmp_path, manifest={"models": manifest}))
    assert report.row.quality.judge == "qwen3-14b"
    assert report.row.quality.judge_source == "dataset"


def test_recall_carries_the_datasets_granularity_label(tmp_path: Path) -> None:
    report = run_score(build(tmp_path))
    assert report.row.retrieval.evidence_unit == "message"


def test_granularity_follows_the_dataset_rather_than_a_default(tmp_path: Path) -> None:
    dataset = {**MANIFEST["dataset"], "evidence_granularity": "session"}
    report = run_score(build(tmp_path, manifest={"dataset": dataset}))
    assert report.row.retrieval.evidence_unit == "session"


def test_recall_is_averaged_over_measurable_questions(tmp_path: Path) -> None:
    report = run_score(build(tmp_path))
    assert report.row.retrieval.recall_at_k == 0.5


def test_retrieval_track_is_reported(tmp_path: Path) -> None:
    report = run_score(build(tmp_path))
    assert report.row.retrieval.track == "exact"


def test_latency_percentiles_are_reported_per_stage(tmp_path: Path) -> None:
    report = run_score(build(tmp_path))
    assert report.row.latency.retrieve_p50_ms is not None
    assert report.row.latency.ingest_p50_ms is not None


def test_harness_cost_is_labelled_measured(tmp_path: Path) -> None:
    report = run_score(build(tmp_path))
    assert report.row.cost.harness_attribution == "measured"
    assert report.row.cost.harness_prompt_tokens == 300
    assert report.row.cost.harness_completion_tokens == 30


def test_system_cost_is_absent_without_gateway_data(tmp_path: Path) -> None:
    """Phase 2 has no gateway, so system-side cost must be None rather than zero."""
    report = run_score(build(tmp_path))
    assert report.row.cost.system_total_tokens is None
    assert report.row.cost.system_attribution == "unavailable"


def test_robustness_counts_retries(tmp_path: Path) -> None:
    report = run_score(build(tmp_path))
    assert report.row.robustness.ingest_retries == 1


def test_robustness_reports_zero_contract_violations_for_a_clean_run(tmp_path: Path) -> None:
    report = run_score(build(tmp_path))
    assert report.row.robustness.contract_violations == 0


def test_report_has_no_composite_score(tmp_path: Path) -> None:
    """ARCH §7: no single composite number; the reader computes any index."""
    payload = read_json(build_and_read(tmp_path))
    assert "composite" not in json.dumps(payload).lower()


def build_and_read(tmp_path: Path) -> Path:
    run_score(build(tmp_path))
    return tmp_path / "report.json"


def test_score_is_pure_and_byte_identical_across_runs(tmp_path: Path) -> None:
    run_dir = build(tmp_path)
    run_score(run_dir)
    first = (run_dir / "report.json").read_bytes()
    (run_dir / "report.json").unlink()
    run_score(run_dir)
    assert (run_dir / "report.json").read_bytes() == first


def test_score_makes_no_network_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P4: score has no network access. Enforced, not asserted in prose."""
    import socket

    def forbid(*args: object, **kwargs: object) -> None:
        raise AssertionError("score must not touch the network")

    monkeypatch.setattr(socket, "socket", forbid)
    monkeypatch.setattr(socket, "create_connection", forbid)
    run_score(build(tmp_path))


def test_score_uses_no_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Determinism: a wall-clock read would make the report unreproducible."""
    import time

    def forbid() -> float:
        raise AssertionError("score must not read the clock")

    monkeypatch.setattr(time, "time", forbid)
    run_score(build(tmp_path))


def test_a_question_never_answered_is_counted_as_incomplete_not_wrong(tmp_path: Path) -> None:
    run_dir = build(tmp_path)
    JsonlArtifact(run_dir / "plan.jsonl").append(
        {
            "kind": "question",
            "id": qid(3),
            "user_id": UID,
            "ctx_id": "c0",
            "q_id": "q3",
            "question": "question 3?",
            "gold": ["x"],
            "evidence_chunk_ids": [],
            "meta": {},
        }
    )
    report = run_score(run_dir)
    assert report.row.quality.questions == 2
    assert report.row.completeness.planned_questions == 3
    assert report.row.completeness.judged_questions == 2


def test_an_mcq_dataset_is_scored_without_a_judge(tmp_path: Path) -> None:
    run_dir = build(tmp_path)
    manifest = read_json(run_dir / "manifest.json")
    manifest["dataset"] = {**manifest["dataset"], "scorer": "mcq", "task_type": "mcq"}
    write_json(run_dir / "manifest.json", manifest)

    plan = tmp_path / "plan.jsonl"
    rows = [json.loads(line) for line in plan.read_text().splitlines()]
    for row in rows:
        if row["kind"] == "question":
            # Each question's gold is the answer that was actually generated for it,
            # so a correct mapping scores 1.0 without any judge involved.
            own = f"answer {row['q_id'][1:]}"
            row["options"] = [f"A. {own}", "B. other"]
            row["gold"] = [own]
    plan.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))

    report = run_score(run_dir)
    assert report.row.quality.accuracy == 1.0
    assert report.row.quality.judge is None


def test_in_process_baselines_are_marked_in_the_report(tmp_path: Path) -> None:
    report = run_score(build(tmp_path), in_process=True)
    assert report.row.in_process is True


def test_a_contract_system_row_is_not_marked_in_process(tmp_path: Path) -> None:
    assert run_score(build(tmp_path)).row.in_process is False


def test_report_records_the_artifact_counts_it_scored(tmp_path: Path) -> None:
    report = run_score(build(tmp_path))
    assert report.row.completeness.ingested_chunks == 1
    assert report.row.completeness.retrieved_questions == 2


def test_system_cost_attribution_follows_the_verification(tmp_path: Path) -> None:
    """A row must not claim `derived` cost when verify could not derive it."""
    run_dir = build(tmp_path)
    report = run_score(run_dir, system_attribution="derived", system_prompt_tokens=120)
    assert report.row.cost.system_attribution == "derived"
    assert report.row.cost.system_prompt_tokens == 120


def test_an_unreliable_boundary_omits_the_per_phase_number(tmp_path: Path) -> None:
    """D6(a): a broken boundary omits the number rather than printing an indefensible one."""
    report = run_score(build(tmp_path), system_attribution="unreliable", system_prompt_tokens=120)
    assert report.row.cost.system_attribution == "unreliable"
    assert report.row.cost.system_prompt_tokens is None


def test_the_unavailable_cost_note_is_dropped_once_cost_is_derived(tmp_path: Path) -> None:
    """A note claiming no gateway would contradict the derived number beside it."""
    report = run_score(build(tmp_path), system_attribution="derived", system_prompt_tokens=5)
    assert not any("no metering gateway" in note for note in report.notes)


def test_the_unavailable_cost_note_is_present_without_a_gateway(tmp_path: Path) -> None:
    assert any("no metering gateway" in note for note in run_score(build(tmp_path)).notes)


def test_ingest_wall_clock_spans_concurrent_chunks_rather_than_summing_them(
    tmp_path: Path,
) -> None:
    """Concurrent adds overlap, so the sum of their latencies is not elapsed time."""
    run_dir = build(tmp_path)
    JsonlArtifact(run_dir / "ingest.jsonl").append(
        {
            "id": f"{UID}:chunk-c1",
            "user_id": UID,
            "chunk_id": "c1",
            "status": "ok",
            "attempts": 1,
            "latency_ms": 100,
            "started_at_ms": 1_000,
            "ended_at_ms": 1_100,
        }
    )
    # Both chunks ran 1000→1100, so ingest took 100ms of wall clock, not 200ms.
    assert run_score(run_dir).row.latency.ingest_wall_clock_ms == 100
