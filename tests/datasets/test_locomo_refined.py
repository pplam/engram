"""The LoCoMo-Refined adapter, against the real upstream shape (§9, D1).

The fixture in `tests/datasets/real_shapes/locomo_sample.json` is a trimmed copy of
`locomo10.json` — same field names, same `dia_id` evidence handles, same category-5
`adversarial_answer` quirk. Trimmed so the test stays fast and needs no download.
"""

import json
from pathlib import Path

import pytest

from orchestrator.datasets import load_dataset

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / "tests" / "datasets" / "real_shapes" / "locomo_sample.json"


@pytest.fixture
def locomo(tmp_path: Path) -> Path:
    """A datasets root holding locomo-refined, pinned to the trimmed sample."""
    import hashlib
    import shutil

    src = REPO / "datasets" / "locomo-refined"
    dest = tmp_path / "locomo-refined"
    dest.mkdir(parents=True)
    shutil.copy(src / "adapter.py", dest / "adapter.py")

    (dest / "data").mkdir()
    payload = SAMPLE.read_bytes()
    (dest / "data" / "locomo10.json").write_bytes(payload)

    manifest = (src / "dataset.yaml").read_text()
    # Repin the hash to the trimmed sample: the loader verifies it, by design.
    real = hashlib.sha256(payload).hexdigest()
    lines = [
        f"  sha256: {real}" if line.strip().startswith("sha256:") else line
        for line in manifest.splitlines()
    ]
    (dest / "dataset.yaml").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_it_loads_one_context_per_conversation(locomo: Path) -> None:
    dataset = load_dataset("locomo-refined", locomo)
    assert len(dataset.contexts) == 1
    assert dataset.contexts[0].ctx_id == "conv-26"


def test_many_questions_share_one_context(locomo: Path) -> None:
    """LoCoMo's shape: ~138 questions per context. The opposite of LongMemEval."""
    dataset = load_dataset("locomo-refined", locomo)
    assert len(dataset.questions) > 1
    assert {q.ctx_id for q in dataset.questions} == {"conv-26"}


def test_evidence_handles_are_dia_ids(locomo: Path) -> None:
    """Evidence is `D<session>:<turn>`, which the adapter must pass through verbatim."""
    dataset = load_dataset("locomo-refined", locomo)
    handles = {h for q in dataset.questions for h in q.evidence_chunk_ids}
    assert handles
    assert all(h.startswith("D") and ":" in h for h in handles), handles


def test_messages_carry_their_dia_id(locomo: Path) -> None:
    """Recall is by ID, so the handle must survive onto the message the chunker sees."""
    dataset = load_dataset("locomo-refined", locomo)
    messages = dataset.contexts[0].messages
    assert all(m.get("dia_id") for m in messages)


def test_an_adversarial_question_uses_its_adversarial_answer(locomo: Path) -> None:
    """Category 5 answers live in `adversarial_answer`; using `answer` would score noise."""
    raw = json.loads(SAMPLE.read_text())
    expected = {
        q["adversarial_answer"]
        for q in raw[0]["qa"]
        if q.get("category") == 5 and q.get("adversarial_answer")
    }
    assert expected, "the fixture must contain an adversarial question to pin this"

    dataset = load_dataset("locomo-refined", locomo)
    adversarial = [q for q in dataset.questions if q.meta.get("category") == "5"]
    assert adversarial
    assert {a.gold[0] for a in adversarial} == expected


def test_the_manifest_pins_an_official_judge_on_a_routable_provider(locomo: Path) -> None:
    """D7: the dataset pins its own judge, and the pin has to name a real provider.

    Upstream's judge is Qwen3-14B; this repo serves the judge from its own endpoint
    instead, so quality here is not upstream's official metric (see dataset.yaml). What
    still must hold is that a judge is pinned at all and that it is reachable — the
    override is the mechanism D7 added, and an unroutable pin fails the run at the judge
    stage. `tests/gateway/test_config.py` checks the routing half against the gateway.
    """
    dataset = load_dataset("locomo-refined", locomo)
    assert dataset.config.official_judge is not None
    assert dataset.config.official_judge.provider


def test_evidence_is_message_granular(locomo: Path) -> None:
    dataset = load_dataset("locomo-refined", locomo)
    assert dataset.config.evidence_granularity == "message"


def test_messages_use_the_internal_contract(locomo: Path) -> None:
    """Chunking reads `content`; an adapter passing upstream field names through crashes.

    The adapter's job is to normalize into the harness's shape, so `role` and `content`
    are required. `dia_id` rides along because recall is compared against it.
    """
    dataset = load_dataset("locomo-refined", locomo)
    message = dataset.contexts[0].messages[0]
    assert message["content"]
    assert message["role"]
    assert message["dia_id"]


def test_the_speaker_is_preserved_in_the_content(locomo: Path) -> None:
    """Who said a thing is part of the memory; dropping it loses answerable detail."""
    dataset = load_dataset("locomo-refined", locomo)
    raw = json.loads(SAMPLE.read_text())
    speaker = raw[0]["conversation"]["session_1"][0]["speaker"]
    assert speaker in dataset.contexts[0].messages[0]["content"]
