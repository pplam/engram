"""Dataset-then-suite precedence for judge and prompt, in one place (D7, §2.9)."""

import shutil
from pathlib import Path

import pytest

from orchestrator.datasets import load_config
from orchestrator.resolution import resolve_answer_prompt, resolve_judge
from orchestrator.suite import Suite, load_suite

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "datasets" / "fixtures"


@pytest.fixture
def suite() -> Suite:
    return load_suite("v1", REPO)


@pytest.fixture
def datasets(tmp_path: Path) -> Path:
    dest = tmp_path / "datasets"
    shutil.copytree(FIXTURES, dest)
    return dest


def test_neither_set_uses_the_suite_for_both(suite: Suite, datasets: Path) -> None:
    config = load_config("fixture_many", datasets)
    judge = resolve_judge(suite, config)
    prompt = resolve_answer_prompt(suite, config)
    assert (judge.id, judge.source) == (suite.models.judge.id, "suite")
    assert prompt.source == "suite"
    assert prompt.text == suite.prompts.answer.text


def test_both_set_uses_the_dataset_for_both(suite: Suite, datasets: Path) -> None:
    config = load_config("fixture_single", datasets)
    judge = resolve_judge(suite, config)
    prompt = resolve_answer_prompt(suite, config)
    assert (judge.id, judge.source) == ("qwen3-14b", "dataset")
    assert prompt.source == "dataset"
    assert "shortest exact answer" in prompt.text


def test_a_dataset_judge_brings_its_own_provider(datasets: Path) -> None:
    """D11: the pair is `(provider, id)`, so an override must replace both halves.

    Keeping the suite's provider under the dataset's model id would name a pair nobody
    pinned, and the gateway would have nowhere to route the call.
    """
    suite = load_suite("v2", REPO)
    judge = resolve_judge(suite, load_config("fixture_single", datasets))
    assert (judge.id, judge.provider) == ("qwen3-14b", "vllm")
    assert suite.models.judge.provider != "vllm"


def test_the_suite_judge_carries_the_suites_provider(datasets: Path) -> None:
    suite = load_suite("v2", REPO)
    judge = resolve_judge(suite, load_config("fixture_many", datasets))
    assert judge.provider == suite.models.judge.provider


def test_judge_only_leaves_the_prompt_on_the_suite(suite: Suite, datasets: Path) -> None:
    path = datasets / "fixture_single" / "dataset.yaml"
    path.write_text(path.read_text().replace("official_prompt: prompts/answer.txt\n", ""))
    config = load_config("fixture_single", datasets)
    assert resolve_judge(suite, config).source == "dataset"
    assert resolve_answer_prompt(suite, config).source == "suite"


def test_prompt_only_leaves_the_judge_on_the_suite(suite: Suite, datasets: Path) -> None:
    path = datasets / "fixture_single" / "dataset.yaml"
    body = path.read_text()
    body = body[: body.index("official_judge:")]
    path.write_text(body)
    config = load_config("fixture_single", datasets)
    assert resolve_answer_prompt(suite, config).source == "dataset"
    assert resolve_judge(suite, config).source == "suite"


def test_resolved_judge_carries_its_temperature(suite: Suite, datasets: Path) -> None:
    assert resolve_judge(suite, load_config("fixture_single", datasets)).temperature == 0


def test_a_dataset_prompt_pointing_nowhere_aborts(suite: Suite, datasets: Path) -> None:
    (datasets / "fixture_single" / "prompts" / "answer.txt").unlink()
    config = load_config("fixture_single", datasets)
    with pytest.raises(FileNotFoundError, match="answer.txt"):
        resolve_answer_prompt(suite, config)


def test_resolved_prompt_carries_a_hash_for_the_manifest(suite: Suite, datasets: Path) -> None:
    from orchestrator.suite import sha256_text

    prompt = resolve_answer_prompt(suite, load_config("fixture_single", datasets))
    assert prompt.sha256 == sha256_text(prompt.text)


def test_suite_sourced_prompt_hash_matches_the_suite_pin(suite: Suite, datasets: Path) -> None:
    prompt = resolve_answer_prompt(suite, load_config("fixture_many", datasets))
    assert prompt.sha256 == suite.prompts.answer.sha256
