"""`plan` is deterministic, writes the manifest first, and --limit stays consistent (§2.5)."""

import shutil
from pathlib import Path

import pytest

from orchestrator.artifacts import JsonlArtifact, read_json
from orchestrator.datasets import load_dataset
from orchestrator.plan_records import ChunkRecord, QuestionRecord, parse_plan_record
from orchestrator.stages.plan import PlanError, run_plan
from orchestrator.suite import load_suite

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "datasets" / "fixtures"


@pytest.fixture
def datasets(tmp_path: Path) -> Path:
    dest = tmp_path / "datasets"
    shutil.copytree(FIXTURES, dest)
    return dest


@pytest.fixture
def out(tmp_path: Path) -> Path:
    return tmp_path / "artifacts" / "run1"


def plan_for(
    datasets: Path,
    out: Path,
    name: str = "fixture_many",
    limit: int | None = None,
    suite: str = "v1",
) -> Path:
    run_plan(
        run_id="run1",
        system="mock",
        dataset=load_dataset(name, datasets),
        suite=load_suite(suite, REPO),
        out_dir=out,
        limit=limit,
    )
    return out


def records(out: Path) -> list[dict[str, object]]:
    return list(JsonlArtifact(out / "plan.jsonl").stream())


def chunks(out: Path) -> list[dict[str, object]]:
    return [r for r in records(out) if r["kind"] == "chunk"]


def questions(out: Path) -> list[dict[str, object]]:
    return [r for r in records(out) if r["kind"] == "question"]


def test_writes_plan_and_manifest(datasets: Path, out: Path) -> None:
    plan_for(datasets, out)
    assert (out / "plan.jsonl").is_file()
    assert (out / "manifest.json").is_file()


def test_manifest_is_written_before_the_plan(datasets: Path, out: Path) -> None:
    plan_for(datasets, out)
    assert (out / "manifest.json").stat().st_mtime_ns <= (out / "plan.jsonl").stat().st_mtime_ns


def test_run_aborts_if_the_manifest_cannot_be_written(datasets: Path, tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    with pytest.raises((PlanError, OSError)):
        plan_for(datasets, blocked / "run1")


def test_no_plan_is_written_when_the_manifest_fails(datasets: Path, tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    with pytest.raises((PlanError, OSError)):
        plan_for(datasets, blocked / "run1")
    assert not (blocked / "run1" / "plan.jsonl").exists()


def test_emits_one_chunk_record_per_chunk(datasets: Path, out: Path) -> None:
    plan_for(datasets, out)
    assert len(chunks(out)) >= 2
    assert all(r["chunk_id"] for r in chunks(out))


def test_emits_one_question_record_per_question(datasets: Path, out: Path) -> None:
    plan_for(datasets, out)
    assert len(questions(out)) == 5


def test_chunk_ids_are_namespaced_by_run_id(datasets: Path, out: Path) -> None:
    plan_for(datasets, out)
    assert all(str(r["id"]).startswith("eval:run1:fixture_many:") for r in records(out))


def test_chunk_record_ids_use_the_chunk_form(datasets: Path, out: Path) -> None:
    plan_for(datasets, out)
    assert all(":chunk-" in str(r["id"]) for r in chunks(out))


def test_question_record_ids_use_the_question_form(datasets: Path, out: Path) -> None:
    plan_for(datasets, out)
    assert all(":q-" in str(r["id"]) for r in questions(out))


def test_question_records_carry_gold_and_evidence(datasets: Path, out: Path) -> None:
    plan_for(datasets, out)
    record = next(r for r in questions(out) if str(r["id"]).endswith(":q-q1"))
    assert record["gold"] == ["Mochi"]
    assert record["evidence_chunk_ids"] == ["m0"]


def test_evidence_handles_are_resolved_to_chunk_ids(datasets: Path, out: Path) -> None:
    """Recall compares against `x_source_ids`, which are chunk ids, not `m0` handles."""
    plan_for(datasets, out)
    record = next(r for r in questions(out) if str(r["id"]).endswith(":q-q1"))
    assert record["evidence_chunks"] == ["c0"]


def test_the_original_dataset_handles_are_kept_alongside_the_expansion(
    datasets: Path, out: Path
) -> None:
    """The dataset's own annotation stays auditable after the mapping."""
    plan_for(datasets, out)
    record = next(r for r in questions(out) if str(r["id"]).endswith(":q-q3"))
    assert record["evidence_chunk_ids"] == ["m1"]
    assert record["evidence_chunks"] == ["c0"]


def test_session_level_evidence_expands_to_every_chunk_of_that_session(
    datasets: Path, out: Path
) -> None:
    dataset = load_dataset("fixture_single", datasets)
    run_plan(
        run_id="r1",
        system="bm25",
        dataset=dataset,
        suite=load_suite("v1", REPO),
        out_dir=out,
    )
    record = next(iter(questions(out)))
    expanded = record["evidence_chunks"]
    assert isinstance(expanded, list)
    assert expanded
    assert all(str(cid).startswith("c") for cid in expanded)


def test_question_records_carry_options_when_present(datasets: Path, out: Path) -> None:
    plan_for(datasets, out)
    record = next(r for r in questions(out) if str(r["id"]).endswith(":q-q5"))
    assert record["options"] == ["A. Lisbon", "B. Berlin"]


def test_records_round_trip_through_their_models(datasets: Path, out: Path) -> None:
    plan_for(datasets, out)
    for raw in records(out):
        parsed = parse_plan_record(raw)
        assert parsed.model_dump(exclude_none=True, mode="json") == raw


def test_chunk_and_question_records_parse_to_their_own_types(datasets: Path, out: Path) -> None:
    plan_for(datasets, out)
    assert isinstance(parse_plan_record(chunks(out)[0]), ChunkRecord)
    assert isinstance(parse_plan_record(questions(out)[0]), QuestionRecord)


def test_same_dataset_and_suite_produce_byte_identical_plans(
    datasets: Path, tmp_path: Path
) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    plan_for(datasets, first)
    plan_for(datasets, second)
    assert (first / "plan.jsonl").read_bytes() == (second / "plan.jsonl").read_bytes()


def test_replanning_into_the_same_directory_adds_nothing(datasets: Path, out: Path) -> None:
    plan_for(datasets, out)
    before = (out / "plan.jsonl").read_bytes()
    plan_for(datasets, out)
    assert (out / "plan.jsonl").read_bytes() == before


def test_limit_truncates_the_questions(datasets: Path, out: Path) -> None:
    plan_for(datasets, out, limit=2)
    assert len(questions(out)) == 2


def test_limit_keeps_only_the_contexts_its_questions_need(datasets: Path, out: Path) -> None:
    """Limiting must not drag in haystacks for questions that were cut."""
    plan_for(datasets, out, limit=1)
    kept = {str(r["ctx_id"]) for r in questions(out)}
    assert {str(r["ctx_id"]) for r in chunks(out)} == kept


def test_limit_never_orphans_a_context(datasets: Path, out: Path) -> None:
    plan_for(datasets, out, limit=4)
    planned = {str(r["ctx_id"]) for r in chunks(out)}
    assert all(str(r["ctx_id"]) in planned for r in questions(out))


def test_limit_larger_than_the_dataset_keeps_everything(datasets: Path, out: Path) -> None:
    plan_for(datasets, out, limit=999)
    assert len(questions(out)) == 5


def test_limit_is_deterministic(datasets: Path, tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    plan_for(datasets, first, limit=3)
    plan_for(datasets, second, limit=3)
    assert (first / "plan.jsonl").read_bytes() == (second / "plan.jsonl").read_bytes()


def test_works_for_a_one_context_per_question_dataset(datasets: Path, out: Path) -> None:
    plan_for(datasets, out, name="fixture_single")
    assert len(questions(out)) == 3
    assert len({str(r["ctx_id"]) for r in chunks(out)}) == 3


def test_limit_on_a_one_context_per_question_dataset_cuts_haystacks(
    datasets: Path, out: Path
) -> None:
    plan_for(datasets, out, name="fixture_single", limit=1)
    assert len(questions(out)) == 1
    assert len({str(r["ctx_id"]) for r in chunks(out)}) == 1


def test_manifest_records_the_pins_a_reader_needs(datasets: Path, out: Path) -> None:
    plan_for(datasets, out)
    manifest = read_json(out / "manifest.json")
    assert manifest["run_id"] == "run1"
    assert manifest["system"]["name"] == "mock"
    assert manifest["suite"]["version"] == "v1"
    assert manifest["suite"]["sha256"]
    assert manifest["dataset"]["name"] == "fixture_many"
    assert manifest["dataset"]["source_sha256"]
    assert manifest["top_k"] == 100
    assert manifest["prompts"]["answer_sha256"]
    assert manifest["prompts"]["judge_sha256"]


def test_manifest_records_the_resolved_judge_and_its_source(datasets: Path, out: Path) -> None:
    plan_for(datasets, out)
    models = read_json(out / "manifest.json")["models"]
    assert models["judge"] == "gpt-4.1-mini"
    assert models["judge_source"] == "suite"


def test_manifest_records_a_dataset_pinned_judge_as_dataset_sourced(
    datasets: Path, out: Path
) -> None:
    plan_for(datasets, out, name="fixture_single")
    models = read_json(out / "manifest.json")["models"]
    assert models["judge"] == "qwen3-14b"
    assert models["judge_source"] == "dataset"


def test_manifest_records_the_provider_serving_each_model(datasets: Path, out: Path) -> None:
    """A model id alone does not say which endpoint answered (D11)."""
    plan_for(datasets, out, suite="v2")
    providers = read_json(out / "manifest.json")["models"]["providers"]
    assert providers == {"chat": "openai", "embedding": "openai", "judge": "openai"}


def test_manifest_records_a_dataset_judges_own_provider(datasets: Path, out: Path) -> None:
    plan_for(datasets, out, name="fixture_single", suite="v2")
    models = read_json(out / "manifest.json")["models"]
    assert (models["judge"], models["providers"]["judge"]) == ("qwen3-14b", "vllm")


def test_manifest_records_no_provider_for_a_legacy_suite(datasets: Path, out: Path) -> None:
    """v1 named bare ids, so there is nothing to record rather than something to invent."""
    plan_for(datasets, out)
    providers = read_json(out / "manifest.json")["models"]["providers"]
    assert providers == {"chat": None, "embedding": None, "judge": None}


def test_manifest_records_the_limit_when_set(datasets: Path, out: Path) -> None:
    plan_for(datasets, out, limit=2)
    assert read_json(out / "manifest.json")["limit"] == 2
