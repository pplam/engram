"""A run records what it did at every stage boundary, and never records a payload.

The second property is the load-bearing one: ARCH forbids logging prompts, messages,
retrieved content, or keys. A log that leaks a question is worse than no log.
"""

import json
import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest

from contract.adapter import ContractViolation, Memory, Message
from orchestrator.models import StubChat
from orchestrator.run import RunRequest, execute_run
from orchestrator.runlog import RUN_LOG
from reference.baselines import build_baseline

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "datasets" / "fixtures"


@pytest.fixture
def datasets(tmp_path: Path) -> Path:
    dest = tmp_path / "datasets"
    shutil.copytree(FIXTURES, dest)
    return dest


def request_for(datasets: Path, tmp_path: Path, system: str = "bm25") -> RunRequest:
    return RunRequest(
        run_id="r1",
        system=system,
        dataset="fixture_many",
        suite="v1",
        artifacts_root=tmp_path / "artifacts",
        datasets_root=datasets,
        repo_root=REPO,
    )


# The reply carries a reasoning field so the payload check below has something to look
# for. `label` itself is a status and may be logged; the judge's prose may not.
JUDGE_REASONING = "the prediction restates the gold answer"


def judge_stub() -> StubChat:
    return StubChat(
        replies=['{"label": "CORRECT", "reasoning": "' + JUDGE_REASONING + '"}'],
        prompt_tokens=7,
        completion_tokens=2,
    )


async def log_of(request: RunRequest, system: str = "bm25") -> str:
    adapter = None if request.is_in_process else build_baseline(system)
    await execute_run(
        request,
        adapter,
        chat=StubChat(replies=["an answer"], prompt_tokens=5, completion_tokens=3),
        judge_chat=judge_stub(),
    )
    return (request.run_dir / RUN_LOG).read_text()


async def test_a_run_writes_a_log_beside_its_artifacts(datasets: Path, tmp_path: Path) -> None:
    request = request_for(datasets, tmp_path)
    assert await log_of(request)


@pytest.mark.parametrize(
    "stage", ["plan", "ingest", "retrieve", "answer", "judge", "verify", "score"]
)
async def test_every_stage_reports_that_it_ran(datasets: Path, tmp_path: Path, stage: str) -> None:
    text = await log_of(request_for(datasets, tmp_path))
    assert stage in text, f"{stage} left no trace in run.log"


async def test_the_log_names_the_run_and_its_configuration(datasets: Path, tmp_path: Path) -> None:
    text = await log_of(request_for(datasets, tmp_path))
    assert "r1" in text
    assert "bm25" in text
    assert "fixture_many" in text


async def test_stage_records_carry_counts(datasets: Path, tmp_path: Path) -> None:
    """A log that says "ingest done" and not how much it ingested cannot answer "is it stuck"."""
    text = await log_of(request_for(datasets, tmp_path))
    ingest = [line for line in text.splitlines() if "ingest" in line and "done" in line]
    assert ingest, text
    assert any(any(ch.isdigit() for ch in line) for line in ingest)


async def test_model_calls_are_logged_with_token_counts(datasets: Path, tmp_path: Path) -> None:
    text = await log_of(request_for(datasets, tmp_path))
    assert "prompt_tokens=5" in text, "the answer stage's model spend is not recorded"
    assert "prompt_tokens=7" in text, "the judge stage's model spend is not recorded"


async def test_the_log_never_carries_a_prompt_or_an_answer(datasets: Path, tmp_path: Path) -> None:
    """IDs, statuses, counts, latencies. Never content."""
    request = request_for(datasets, tmp_path)
    text = await log_of(request)
    assert "an answer" not in text, "a generated answer reached the log"
    assert JUDGE_REASONING not in text, "the judge's reply body reached the log"
    for line in (request.run_dir / "plan.jsonl").read_text().splitlines():
        assert line not in text
    # Every question the plan carries, and every memory the system returned.
    for row in (request.run_dir / "retrieve.jsonl").read_text().splitlines():
        query = json.loads(row)["query"]
        assert query not in text, "a question reached the log"


class BrokenAdapter:
    """Violates the contract on `search`, so the run must fail loudly (ARCH §6)."""

    async def add(self, user_id: str, messages: Sequence[Message], chunk_id: str) -> None:
        return None

    async def search(self, user_id: str, query: str, top_k: int) -> Sequence[Memory]:
        raise ContractViolation("search returned a memory with empty content")


async def test_a_failing_run_records_the_failure(datasets: Path, tmp_path: Path) -> None:
    request = request_for(datasets, tmp_path)
    with pytest.raises(ContractViolation):
        await execute_run(
            request,
            BrokenAdapter(),
            chat=StubChat(replies=["an answer"]),
            judge_chat=judge_stub(),
        )
    text = (request.run_dir / RUN_LOG).read_text()
    assert "failed" in text.lower()
    assert "ContractViolation" in text


class TestModelIdentity:
    """Every model call names the model that served it (ARCH §12).

    A model id is not content, and it is the one field that makes a per-call line
    attributable: a dataset may override the judge, so `answer` and `judge` in the same
    run can be served by different models. Without it, a slow or refusing call cannot be
    pinned to a model after the fact.
    """

    async def test_the_run_header_names_the_resolved_models(
        self, datasets: Path, tmp_path: Path
    ) -> None:
        """`bench logs` should say what a run's models were without reading the manifest.

        Its own line rather than `run start`: that one is written before the suite is
        loaded, so it cannot name a model the run has not resolved yet.
        """
        text = await log_of(request_for(datasets, tmp_path))
        header = next(line for line in text.splitlines() if "run models" in line)
        assert "chat_model=gpt-4o-mini" in header
        assert "judge_model=gpt-4.1-mini" in header
        assert "embedding_model=text-embedding-3-small" in header

    async def test_each_answer_call_names_its_model(self, datasets: Path, tmp_path: Path) -> None:
        text = await log_of(request_for(datasets, tmp_path))
        calls = [line for line in text.splitlines() if " answer id=" in line]
        assert calls, text
        assert all("model=gpt-4o-mini" in line for line in calls)

    async def test_each_judge_call_names_its_model(self, datasets: Path, tmp_path: Path) -> None:
        """The judge differs from the chat model, so a shared label would hide which ran."""
        text = await log_of(request_for(datasets, tmp_path))
        calls = [line for line in text.splitlines() if " judge id=" in line]
        assert calls, text
        assert all("model=gpt-4.1-mini" in line for line in calls)

    async def test_a_dataset_judge_override_is_the_one_logged(self, tmp_path: Path) -> None:
        """The *resolved* judge is what ran, so it is what the log must name (D11)."""
        dest = tmp_path / "datasets"
        shutil.copytree(FIXTURES, dest)
        request = RunRequest(
            run_id="r1",
            system="bm25",
            dataset="fixture_single",
            suite="v1",
            artifacts_root=tmp_path / "artifacts",
            datasets_root=dest,
            repo_root=REPO,
        )
        text = await log_of(request)
        assert "judge_model=vllm/qwen3-14b" in text
        assert "gpt-4.1-mini" not in text, "the suite's judge was logged, not the resolved one"

    async def test_the_model_id_does_not_bring_a_payload_with_it(
        self, datasets: Path, tmp_path: Path
    ) -> None:
        """Naming the model must not become an excuse to log the call around it."""
        text = await log_of(request_for(datasets, tmp_path))
        assert "an answer" not in text
        assert JUDGE_REASONING not in text
