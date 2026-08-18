"""`verify` publishes adversarial checks beside the report (ARCH §8, §3.3).

Severities differ by design: leakage invalidates a run, unmetered model use withholds
a row, rank instability is reported as variance rather than failure.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from orchestrator.artifacts import JsonlArtifact, write_json
from orchestrator.metrics import Usage
from orchestrator.verify import (
    Verification,
    check_phase_ordering,
    check_unmetered_model_use,
    kendall_tau,
    rank_stability,
    run_verify,
)

UID = "eval:r1:d:c0"


def ingest_row(chunk: str, started: int, ended: int) -> dict[str, Any]:
    return {
        "id": f"{UID}:chunk-{chunk}",
        "user_id": UID,
        "chunk_id": chunk,
        "status": "ok",
        "attempts": 1,
        "latency_ms": 10,
        "started_at_ms": started,
        "ended_at_ms": ended,
    }


def retrieve_row(q: str, started: int, ended: int) -> dict[str, Any]:
    return {
        "id": f"{UID}:q-{q}",
        "user_id": UID,
        "query": "why?",
        "top_k": 5,
        "attempts": 1,
        "latency_ms": 10,
        "started_at_ms": started,
        "ended_at_ms": ended,
        "data": [],
    }


def test_phases_that_do_not_overlap_pass() -> None:
    """The D6(a) invariant: ingest is fully drained before retrieve opens."""
    result = check_phase_ordering(
        [ingest_row("c0", 100, 200), ingest_row("c1", 150, 250)],
        [retrieve_row("q1", 260, 300)],
    )
    assert result.passed


def test_a_retrieve_starting_before_ingest_ends_is_caught() -> None:
    """Overlap makes system-side per-phase cost silently wrong, so it must be loud."""
    result = check_phase_ordering(
        [ingest_row("c0", 100, 400)],
        [retrieve_row("q1", 300, 500)],
    )
    assert not result.passed
    assert "overlap" in result.detail


def test_phase_ordering_is_unmeasurable_without_timestamps() -> None:
    """An in-process baseline has no ingest phase at all; that is not a violation."""
    result = check_phase_ordering([], [retrieve_row("q1", 100, 200)])
    assert result.passed
    assert result.detail == "no ingest phase to order against"


def test_gateway_activity_matching_the_phases_passes() -> None:
    result = check_unmetered_model_use(
        answered=2, judged=2, usage={"answer": Usage(requests=2), "judge": Usage(requests=2)}
    )
    assert result.passed


def test_a_phase_with_answers_but_no_gateway_calls_is_flagged() -> None:
    """A system calling a provider directly is the whole reason verify exists."""
    result = check_unmetered_model_use(answered=5, judged=5, usage={})
    assert not result.passed
    assert "answer" in result.detail


def test_more_gateway_calls_than_answers_is_not_a_failure() -> None:
    """Retries legitimately spend more calls than there are records."""
    result = check_unmetered_model_use(
        answered=1, judged=1, usage={"answer": Usage(requests=3), "judge": Usage(requests=1)}
    )
    assert result.passed


def test_identical_rankings_are_perfectly_stable() -> None:
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_a_reversed_ranking_is_perfectly_unstable() -> None:
    assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0


def test_one_swap_is_partially_stable() -> None:
    assert kendall_tau(["a", "b", "c"], ["b", "a", "c"]) == pytest.approx(1 / 3)


def test_tau_needs_two_comparable_items() -> None:
    assert kendall_tau(["a"], ["a"]) is None


def test_only_items_in_both_rankings_are_compared() -> None:
    """A system may return a different set on a repeat; rank what overlaps."""
    assert kendall_tau(["a", "b"], ["b", "a", "z"]) == -1.0


def test_rank_stability_is_reported_as_variance_not_a_failure() -> None:
    result = rank_stability([(["a", "b"], ["b", "a"])])
    assert result.passed
    assert result.tau == -1.0


def test_rank_stability_without_repeats_is_unmeasured() -> None:
    result = rank_stability([])
    assert result.passed
    assert result.tau is None


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    write_json(
        tmp_path / "manifest.json",
        {"run_id": "r1", "system": {"name": "bm25"}, "models": {"chat": "m", "judge": "m"}},
    )
    ingest = JsonlArtifact(tmp_path / "ingest.jsonl")
    ingest.append(ingest_row("c0", 100, 200))
    retrieve = JsonlArtifact(tmp_path / "retrieve.jsonl")
    retrieve.append(retrieve_row("q1", 300, 400))
    answer = JsonlArtifact(tmp_path / "answer.jsonl")
    answer.append({"id": f"{UID}:q-q1", "generated_answer": "x"})
    judge = JsonlArtifact(tmp_path / "judge.jsonl")
    judge.append({"id": f"{UID}:q-q1", "label": "CORRECT", "is_correct": True})
    return tmp_path


def test_verify_writes_its_own_artifact(run_dir: Path) -> None:
    run_verify(run_dir, usage={"answer": Usage(requests=1), "judge": Usage(requests=1)})
    assert (run_dir / "verify.json").is_file()


def test_a_clean_run_is_not_invalidated(run_dir: Path) -> None:
    result = run_verify(run_dir, usage={"answer": Usage(requests=1), "judge": Usage(requests=1)})
    assert result.run_valid
    assert not result.row_withheld


def test_unmetered_model_use_withholds_the_row(run_dir: Path) -> None:
    result = run_verify(run_dir, usage={})
    assert result.row_withheld
    assert "unmetered_model_use" in result.flags


def test_an_overlapping_phase_boundary_marks_cost_unreliable(tmp_path: Path) -> None:
    write_json(tmp_path / "manifest.json", {"run_id": "r1", "system": {"name": "bm25"}})
    JsonlArtifact(tmp_path / "ingest.jsonl").append(ingest_row("c0", 100, 400))
    JsonlArtifact(tmp_path / "retrieve.jsonl").append(retrieve_row("q1", 300, 500))
    result = run_verify(tmp_path, usage={})
    assert result.cost_attribution == "unreliable"


def test_a_contended_run_records_unreliable_cost(run_dir: Path) -> None:
    """3.5.1's override breaks D6(a) by construction, so it cannot claim clean cost."""
    result = run_verify(
        run_dir, usage={"answer": Usage(requests=1), "judge": Usage(requests=1)}, contended=True
    )
    assert result.cost_attribution == "unreliable"


def test_leakage_invalidates_the_run_rather_than_scoring_zero(run_dir: Path) -> None:
    result = run_verify(
        run_dir,
        usage={"answer": Usage(requests=1), "judge": Usage(requests=1)},
        leaked=True,
    )
    assert not result.run_valid
    assert "cross_user_leakage" in result.flags


def test_verify_json_round_trips(run_dir: Path) -> None:
    run_verify(run_dir, usage={"answer": Usage(requests=1), "judge": Usage(requests=1)})
    raw = json.loads((run_dir / "verify.json").read_text())
    assert Verification.model_validate(raw).verify_version == 1


def test_without_a_gateway_metering_is_unmeasurable_not_violated(run_dir: Path) -> None:
    """No gateway means no counters to reconcile; that must not read as a bypass."""
    result = run_verify(run_dir, usage={}, metered=False)
    assert "unmetered_model_use" not in result.flags
    assert not result.row_withheld


def test_with_a_gateway_an_absent_counter_is_a_bypass(run_dir: Path) -> None:
    result = run_verify(run_dir, usage={}, metered=True)
    assert "unmetered_model_use" in result.flags


def test_metering_defaults_to_on_so_a_bypass_is_never_missed_by_omission(run_dir: Path) -> None:
    assert "unmetered_model_use" in run_verify(run_dir, usage={}).flags


def test_an_unmetered_run_cannot_claim_derived_cost(run_dir: Path) -> None:
    """Without a gateway there is no counter to derive from, so nothing is derived."""
    assert run_verify(run_dir, usage={}, metered=False).cost_attribution == "unavailable"


def test_a_metered_clean_run_derives_cost(run_dir: Path) -> None:
    result = run_verify(
        run_dir, usage={"answer": Usage(requests=1), "judge": Usage(requests=1)}, metered=True
    )
    assert result.cost_attribution == "derived"
