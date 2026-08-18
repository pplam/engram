"""Dataset plugin loading: a dataset is a directory, hash-pinned (§2.3, ARCH §9).

The two fixture datasets have opposite shapes on purpose — many questions per
context, and one context per question. A loader that only handles the first bakes
in an assumption that breaks on LongMemEval-S.
"""

import shutil
from pathlib import Path

import pytest

from orchestrator.datasets import (
    Context,
    DatasetError,
    Question,
    load_dataset,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A writable copy of the fixture dataset tree."""
    dest = tmp_path / "datasets"
    shutil.copytree(FIXTURES, dest)
    return dest


def test_loads_a_many_questions_per_context_dataset(root: Path) -> None:
    loaded = load_dataset("fixture_many", root)
    assert len(loaded.contexts) == 2
    assert len(loaded.questions) == 5


def test_loads_a_one_context_per_question_dataset(root: Path) -> None:
    loaded = load_dataset("fixture_single", root)
    assert len(loaded.contexts) == 3
    assert len(loaded.questions) == 3


def test_every_question_points_at_a_known_context(root: Path) -> None:
    for name in ("fixture_many", "fixture_single"):
        loaded = load_dataset(name, root)
        known = {c.ctx_id for c in loaded.contexts}
        assert all(q.ctx_id in known for q in loaded.questions)


def test_context_carries_messages_in_order(root: Path) -> None:
    ctx = load_dataset("fixture_many", root).contexts[0]
    assert [m["content"] for m in ctx.messages] == [
        "I adopted a tabby cat named Mochi in March 2024.",
        "Mochi sleeps on the windowsill every afternoon.",
        "I moved to Lisbon in June 2024 and took Mochi with me.",
    ]


def test_question_carries_gold_and_evidence(root: Path) -> None:
    q = next(q for q in load_dataset("fixture_many", root).questions if q.q_id == "q1")
    assert q.gold == ("Mochi",)
    assert q.evidence_chunk_ids == ("m0",)


def test_question_options_are_present_only_for_choice_tasks(root: Path) -> None:
    questions = {q.q_id: q for q in load_dataset("fixture_many", root).questions}
    assert questions["q1"].options is None
    assert questions["q5"].options == ("A. Lisbon", "B. Berlin")


def test_manifest_fields_are_exposed_for_the_run_manifest(root: Path) -> None:
    loaded = load_dataset("fixture_many", root)
    assert loaded.config.name == "fixture_many"
    assert loaded.config.version == "1.0"
    assert loaded.config.task_type == "open_qa"
    assert loaded.config.scorer == "judge_binary"
    assert loaded.config.supports_recall is True
    assert loaded.config.source.sha256


def test_evidence_granularity_is_declared_per_dataset(root: Path) -> None:
    """Recall carries a unit label that follows the dataset, never a hardcoded default."""
    assert load_dataset("fixture_many", root).config.evidence_granularity == "message"
    assert load_dataset("fixture_single", root).config.evidence_granularity == "session"


def test_official_prompt_and_judge_are_read_when_declared(root: Path) -> None:
    config = load_dataset("fixture_single", root).config
    assert config.official_prompt == "prompts/answer.txt"
    assert config.official_judge is not None
    assert config.official_judge.id == "qwen3-14b"


def test_official_overrides_are_none_when_absent(root: Path) -> None:
    config = load_dataset("fixture_many", root).config
    assert config.official_prompt is None
    assert config.official_judge is None


def test_source_hash_mismatch_aborts(root: Path) -> None:
    (root / "fixture_many" / "data" / "corpus.json").write_text('{"contexts": []}')
    with pytest.raises(DatasetError, match="sha256"):
        load_dataset("fixture_many", root)


def test_source_hash_mismatch_names_the_dataset(root: Path) -> None:
    (root / "fixture_many" / "data" / "corpus.json").write_text('{"contexts": []}')
    with pytest.raises(DatasetError, match="fixture_many"):
        load_dataset("fixture_many", root)


def test_missing_source_file_aborts(root: Path) -> None:
    (root / "fixture_many" / "data" / "corpus.json").unlink()
    with pytest.raises(DatasetError, match="corpus.json"):
        load_dataset("fixture_many", root)


def test_unknown_dataset_aborts(root: Path) -> None:
    with pytest.raises(DatasetError, match="absent"):
        load_dataset("absent", root)


def test_missing_adapter_aborts(root: Path) -> None:
    (root / "fixture_many" / "adapter.py").unlink()
    with pytest.raises(DatasetError, match="adapter.py"):
        load_dataset("fixture_many", root)


def test_adapter_without_load_aborts(root: Path) -> None:
    (root / "fixture_many" / "adapter.py").write_text("x = 1\n")
    with pytest.raises(DatasetError, match="load"):
        load_dataset("fixture_many", root)


def test_name_mismatch_between_directory_and_yaml_aborts(root: Path) -> None:
    path = root / "fixture_many" / "dataset.yaml"
    path.write_text(path.read_text().replace("name: fixture_many", "name: other"))
    with pytest.raises(DatasetError, match="name"):
        load_dataset("fixture_many", root)


def test_unknown_task_type_aborts(root: Path) -> None:
    path = root / "fixture_many" / "dataset.yaml"
    path.write_text(path.read_text().replace("task_type: open_qa", "task_type: telepathy"))
    with pytest.raises(DatasetError, match="task_type"):
        load_dataset("fixture_many", root)


def test_loading_is_deterministic(root: Path) -> None:
    first, second = load_dataset("fixture_many", root), load_dataset("fixture_many", root)
    assert first.contexts == second.contexts
    assert first.questions == second.questions


def test_records_are_frozen(root: Path) -> None:
    loaded = load_dataset("fixture_many", root)
    with pytest.raises((AttributeError, TypeError)):
        loaded.contexts[0].ctx_id = "other"  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        loaded.questions[0].q_id = "other"  # type: ignore[misc]


def test_returns_the_declared_record_types(root: Path) -> None:
    loaded = load_dataset("fixture_single", root)
    assert all(isinstance(c, Context) for c in loaded.contexts)
    assert all(isinstance(q, Question) for q in loaded.questions)
