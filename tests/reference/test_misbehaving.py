"""Mocks that violate on purpose (§3.3).

Three distinct violations with three distinct outcomes: bypassing the gateway
withholds the row, unstable ranking is reported as variance, and asking for a
non-pinned model fails the run loudly.
"""

from pathlib import Path

import httpx
import pytest

from orchestrator.artifacts import JsonlArtifact, write_json
from orchestrator.metrics import Usage
from orchestrator.models import ModelError, OpenAiCompatibleChat
from orchestrator.verify import rank_stability, run_verify

UID = "eval:r1:d:c0"


def manifest(tmp_path: Path) -> None:
    write_json(tmp_path / "manifest.json", {"run_id": "r1", "system": {"name": "rogue"}})


def test_a_system_bypassing_the_gateway_is_caught(tmp_path: Path) -> None:
    """Phase 3's exit criterion: answers exist but the gateway metered nothing."""
    manifest(tmp_path)
    JsonlArtifact(tmp_path / "answer.jsonl").append({"id": f"{UID}:q-q1", "generated_answer": "x"})
    JsonlArtifact(tmp_path / "judge.jsonl").append({"id": f"{UID}:q-q1", "label": "CORRECT"})

    result = run_verify(tmp_path, usage={})
    assert "unmetered_model_use" in result.flags
    assert result.row_withheld


def test_a_row_is_withheld_rather_than_scored(tmp_path: Path) -> None:
    """An unverifiable row is not a low score; it is a number we decline to publish."""
    manifest(tmp_path)
    JsonlArtifact(tmp_path / "answer.jsonl").append({"id": f"{UID}:q-q1", "generated_answer": "x"})
    result = run_verify(tmp_path, usage={})
    assert result.run_valid, "bypassing the gateway is not the same as leaking data"
    assert result.row_withheld


def test_an_unstable_ranker_reports_variance_and_still_passes() -> None:
    """A black box cannot be seeded, so instability is measured, not punished."""
    result = rank_stability([(["a", "b", "c"], ["c", "a", "b"]), (["a", "b"], ["b", "a"])])
    assert result.passed
    assert result.tau is not None
    assert result.tau < 1.0


def test_a_stable_ranker_reports_perfect_agreement() -> None:
    result = rank_stability([(["a", "b", "c"], ["a", "b", "c"])])
    assert result.tau == 1.0


async def test_a_non_pinned_model_fails_the_run_loudly() -> None:
    """`allowed_models` is the pinning mechanism; a 403 must not score as zero."""

    def blocked(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": "model_blocked"}})

    chat = OpenAiCompatibleChat(
        httpx.AsyncClient(transport=httpx.MockTransport(blocked), base_url="http://gw.invalid"),
        api_key="sk-test",
    )
    with pytest.raises(ModelError, match="model_blocked"):
        await chat.chat(
            [{"role": "user", "content": "hi"}], model="forbidden-model", temperature=0.0
        )


async def test_a_pinned_model_is_allowed_through() -> None:
    def allowed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    chat = OpenAiCompatibleChat(
        httpx.AsyncClient(transport=httpx.MockTransport(allowed), base_url="http://gw.invalid"),
        api_key="sk-test",
    )
    completion = await chat.chat(
        [{"role": "user", "content": "hi"}], model="gpt-4o-mini", temperature=0.0
    )
    assert completion.text == "ok"


def test_a_leaking_system_invalidates_the_run_but_a_bypass_does_not(tmp_path: Path) -> None:
    """The two severities must stay distinct (ARCH §8)."""
    manifest(tmp_path)
    JsonlArtifact(tmp_path / "answer.jsonl").append({"id": f"{UID}:q-q1", "generated_answer": "x"})

    leaking = run_verify(tmp_path, usage={"answer": Usage(requests=1)}, leaked=True)
    assert not leaking.run_valid

    bypassing = run_verify(tmp_path, usage={})
    assert bypassing.run_valid
