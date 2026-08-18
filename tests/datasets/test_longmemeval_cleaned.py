"""The LongMemEval-S Cleaned adapter, against the real upstream shape (§9, D1).

The opposite shape to LoCoMo: one private haystack per question, and evidence annotated
at session level. Both are load-bearing for the plugin API — ARCH §9 picked these two
precisely because a harness that assumes either shape breaks on the other.

The fixture is a trimmed copy of `longmemeval_s_cleaned.json` (the real file is 277 MB),
keeping the gold sessions plus distractors so recall has something to get wrong.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from orchestrator.datasets import load_dataset

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / "tests" / "datasets" / "real_shapes" / "longmemeval_sample.json"


@pytest.fixture
def longmemeval(tmp_path: Path) -> Path:
    """A datasets root holding longmemeval-cleaned, pinned to the trimmed sample."""
    src = REPO / "datasets" / "longmemeval-cleaned"
    dest = tmp_path / "longmemeval-cleaned"
    dest.mkdir(parents=True)
    shutil.copy(src / "adapter.py", dest / "adapter.py")

    (dest / "data").mkdir()
    payload = SAMPLE.read_bytes()
    (dest / "data" / "longmemeval_s_cleaned.json").write_bytes(payload)

    manifest = (src / "dataset.yaml").read_text()
    real = hashlib.sha256(payload).hexdigest()
    lines = [
        f"  sha256: {real}" if line.strip().startswith("sha256:") else line
        for line in manifest.splitlines()
    ]
    (dest / "dataset.yaml").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_each_question_gets_its_own_private_haystack(longmemeval: Path) -> None:
    """The defining shape: one context per question, never shared (ARCH §9)."""
    dataset = load_dataset("longmemeval-cleaned", longmemeval)
    assert len(dataset.contexts) == len(dataset.questions)
    assert len({c.ctx_id for c in dataset.contexts}) == len(dataset.contexts)


def test_a_question_points_at_its_own_context(longmemeval: Path) -> None:
    dataset = load_dataset("longmemeval-cleaned", longmemeval)
    contexts = {c.ctx_id for c in dataset.contexts}
    assert all(q.ctx_id in contexts for q in dataset.questions)


def test_evidence_handles_are_session_ids(longmemeval: Path) -> None:
    """Gold is `answer_session_ids`, which must survive as the evidence handle."""
    raw = json.loads(SAMPLE.read_text())
    expected = set(raw[0]["answer_session_ids"])

    dataset = load_dataset("longmemeval-cleaned", longmemeval)
    first = next(q for q in dataset.questions if q.q_id == raw[0]["question_id"])
    assert set(first.evidence_chunk_ids) == expected


def test_messages_carry_the_session_they_came_from(longmemeval: Path) -> None:
    """A gold session expands to every chunk derived from it, so the id must ride along."""
    dataset = load_dataset("longmemeval-cleaned", longmemeval)
    messages = dataset.contexts[0].messages
    assert all(m.get("session_id") for m in messages)


def test_the_session_id_is_the_upstream_one_not_a_position(longmemeval: Path) -> None:
    """Recall compares against `answer_session_ids`, so renumbering would break matching."""
    raw = json.loads(SAMPLE.read_text())
    dataset = load_dataset("longmemeval-cleaned", longmemeval)
    first = next(c for c in dataset.contexts if c.ctx_id == raw[0]["question_id"])
    assert {m["session_id"] for m in first.messages} == set(raw[0]["haystack_session_ids"])


def test_messages_use_the_internal_contract(longmemeval: Path) -> None:
    dataset = load_dataset("longmemeval-cleaned", longmemeval)
    message = dataset.contexts[0].messages[0]
    assert message["content"]
    assert message["role"] in ("user", "assistant")


def test_evidence_is_session_granular(longmemeval: Path) -> None:
    """A coarser unit is a systematically easier recall target, so it must be labelled."""
    dataset = load_dataset("longmemeval-cleaned", longmemeval)
    assert dataset.config.evidence_granularity == "session"


def test_the_question_type_is_kept_for_per_category_reporting(longmemeval: Path) -> None:
    raw = json.loads(SAMPLE.read_text())
    dataset = load_dataset("longmemeval-cleaned", longmemeval)
    first = next(q for q in dataset.questions if q.q_id == raw[0]["question_id"])
    assert first.meta["question_type"] == raw[0]["question_type"]
