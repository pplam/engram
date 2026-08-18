"""Suite loading: the pinned world, with prompt drift detection (§2.1)."""

import hashlib
from pathlib import Path

import pytest

from orchestrator.suite import CURRENT_SUITE, LEGACY, SuiteError, load_suite

BODY = """
suite: v1
models:
  chat: {id: gpt-4o-mini, temperature: 0, seed: 7}
  embedding: {id: text-embedding-3-small, dimensions: 1536}
  judge: {id: gpt-4.1-mini, temperature: 0}
retrieval:
  top_k: 100
chunking:
  max_messages_per_chunk: 20
  max_words_per_chunk: 2000
  boundary: message_or_sentence
prompts:
  answer: {ref: prompts/answer/v1.txt, sha256: "ANSWER_HASH"}
  judge: {ref: prompts/judge/v1.txt, sha256: "JUDGE_HASH"}
limits:
  ingest: {workers: 16, timeout_s: 1200, attempts: 6}
  retrieve: {workers: 8, timeout_s: 1200, attempts: 6}
  retry_on: [408, 429, 500, 502, 503, 504]
  no_retry_on: [400, 401, 403, 404, 422]
"""


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A tree with a valid suite file and both prompts, hashes correct."""
    answer, judge = "Answer using only the memories.\n", "Judge the answer.\n"
    for rel, text in (("prompts/answer/v1.txt", answer), ("prompts/judge/v1.txt", judge)):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    body = BODY.replace("ANSWER_HASH", sha(answer)).replace("JUDGE_HASH", sha(judge))
    (tmp_path / "suites").mkdir()
    (tmp_path / "suites" / "v1.yaml").write_text(body)
    return tmp_path


def test_loads_the_pinned_models(root: Path) -> None:
    suite = load_suite("v1", root)
    assert suite.models.chat.id == "gpt-4o-mini"
    assert suite.models.chat.temperature == 0
    assert suite.models.chat.seed == 7
    assert suite.models.embedding.id == "text-embedding-3-small"
    assert suite.models.judge.id == "gpt-4.1-mini"


def test_loads_retrieval_and_chunking_pins(root: Path) -> None:
    suite = load_suite("v1", root)
    assert suite.retrieval.top_k == 100
    assert suite.chunking.max_messages_per_chunk == 20
    assert suite.chunking.max_words_per_chunk == 2000
    assert suite.chunking.boundary == "message_or_sentence"


def test_loads_limits_including_retry_policy(root: Path) -> None:
    suite = load_suite("v1", root)
    assert suite.limits.ingest.workers == 16
    assert suite.limits.retrieve.attempts == 6
    assert 429 in suite.limits.retry_on
    assert 400 in suite.limits.no_retry_on


def test_resolves_prompt_text_from_the_ref(root: Path) -> None:
    suite = load_suite("v1", root)
    assert suite.prompts.answer.text.startswith("Answer using only")
    assert suite.prompts.judge.text.startswith("Judge the answer")


def test_prompt_hash_mismatch_aborts_naming_the_file(root: Path) -> None:
    (root / "prompts/answer/v1.txt").write_text("drifted wording\n")
    with pytest.raises(SuiteError, match="answer/v1.txt"):
        load_suite("v1", root)


def test_prompt_hash_mismatch_reports_both_hashes(root: Path) -> None:
    (root / "prompts/answer/v1.txt").write_text("drifted wording\n")
    with pytest.raises(SuiteError, match=sha("drifted wording\n")[:12]):
        load_suite("v1", root)


def test_missing_prompt_file_aborts(root: Path) -> None:
    (root / "prompts/judge/v1.txt").unlink()
    with pytest.raises(SuiteError, match="judge/v1.txt"):
        load_suite("v1", root)


def test_unknown_suite_version_aborts(root: Path) -> None:
    with pytest.raises(SuiteError, match="v9.yaml"):
        load_suite("v9", root)


def test_suite_field_must_match_the_requested_version(root: Path) -> None:
    path = root / "suites" / "v1.yaml"
    path.write_text(path.read_text().replace("suite: v1", "suite: v2"))
    with pytest.raises(SuiteError, match="declares suite"):
        load_suite("v1", root)


def test_missing_required_section_names_the_field(root: Path) -> None:
    path = root / "suites" / "v1.yaml"
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("retrieval")]
    path.write_text("\n".join(ln for ln in lines if "top_k" not in ln))
    with pytest.raises(SuiteError, match="retrieval"):
        load_suite("v1", root)


def test_unknown_key_is_rejected_so_typos_cannot_pass_silently(root: Path) -> None:
    path = root / "suites" / "v1.yaml"
    path.write_text(path.read_text() + "\nunexpected_key: 1\n")
    with pytest.raises(SuiteError, match="unexpected_key"):
        load_suite("v1", root)


def test_suite_carries_its_own_content_hash_for_the_manifest(root: Path) -> None:
    suite = load_suite("v1", root)
    assert suite.sha256 == sha((root / "suites" / "v1.yaml").read_text())


def test_suite_is_frozen(root: Path) -> None:
    suite = load_suite("v1", root)
    with pytest.raises(ValueError, match="frozen|immutable"):
        suite.retrieval.top_k = 5


def test_the_repo_suite_v1_loads_and_verifies() -> None:
    """The committed suites/v1.yaml must be valid against its own prompt hashes."""
    suite = load_suite("v1", Path(__file__).resolve().parents[2])
    assert suite.suite == "v1"
    assert suite.retrieval.top_k > 0


# --- Providers (D11) -------------------------------------------------------------
#
# A pinned model is `(provider, id)` from v2 on. v1 names bare ids and must keep
# loading, because a published row names its suite version and a version's meaning
# cannot be edited afterwards.

V2_BODY = """
suite: v2
models:
  chat: {id: gpt-4o-mini, provider: openai, temperature: 0, seed: 7}
  embedding: {id: text-embedding-3-small, provider: openai, dimensions: 1536}
  judge: {id: qwen3-14b, provider: vllm, temperature: 0}
retrieval:
  top_k: 100
chunking:
  max_messages_per_chunk: 20
  max_words_per_chunk: 2000
  boundary: message_or_sentence
prompts:
  answer: {ref: prompts/answer/v1.txt, sha256: "ANSWER_HASH"}
  judge: {ref: prompts/judge/v1.txt, sha256: "JUDGE_HASH"}
limits:
  ingest: {workers: 16, timeout_s: 1200, attempts: 6}
  retrieve: {workers: 8, timeout_s: 1200, attempts: 6}
"""


def write_v2(root: Path, body: str) -> Path:
    answer = (root / "prompts/answer/v1.txt").read_text()
    judge = (root / "prompts/judge/v1.txt").read_text()
    path = root / "suites" / "v2.yaml"
    path.write_text(body.replace("ANSWER_HASH", sha(answer)).replace("JUDGE_HASH", sha(judge)))
    return path


def test_a_suite_after_v1_loads_each_models_provider(root: Path) -> None:
    write_v2(root, V2_BODY)
    suite = load_suite("v2", root)
    assert suite.models.chat.provider == "openai"
    assert suite.models.embedding.provider == "openai"
    assert suite.models.judge.provider == "vllm"


@pytest.mark.parametrize("role", ["chat", "embedding", "judge"])
def test_a_model_without_a_provider_is_rejected_rather_than_defaulted(
    root: Path, role: str
) -> None:
    """A silent `openai` default is how a run measures a model nobody chose (D11)."""
    stripped = "\n".join(
        line.replace(", provider: openai", "").replace(", provider: vllm", "")
        if line.strip().startswith(f"{role}:")
        else line
        for line in V2_BODY.splitlines()
    )
    write_v2(root, stripped)
    with pytest.raises(SuiteError, match=f"{role}.*provider|provider.*{role}"):
        load_suite("v2", root)


def test_v1_is_the_only_version_allowed_to_name_bare_ids(root: Path) -> None:
    suite = load_suite("v1", root)
    assert suite.models.chat.provider is None


def test_a_suite_after_v1_may_not_carry_a_retry_status_list(root: Path) -> None:
    """Retry is keyed on the exception type an adapter raises, not on a status code."""
    write_v2(root, V2_BODY + "  retry_on: [429, 503]\n")
    with pytest.raises(SuiteError, match="retry_on"):
        load_suite("v2", root)


def test_the_current_suite_is_never_a_legacy_one() -> None:
    """The default has to be held to today's rules, or the exemptions become the norm."""
    assert CURRENT_SUITE not in LEGACY


def test_the_current_suite_exists_and_loads() -> None:
    suite = load_suite(CURRENT_SUITE, Path(__file__).resolve().parents[2])
    assert suite.suite == CURRENT_SUITE


def test_the_repo_suite_v2_loads_and_verifies() -> None:
    """The committed suites/v2.yaml must be valid against its own prompt hashes."""
    suite = load_suite("v2", Path(__file__).resolve().parents[2])
    assert suite.suite == "v2"
    assert suite.models.chat.provider
    assert suite.models.judge.provider
    assert not suite.limits.retry_on


# --- Extraction token budget ----------------------------------------------------
#
# mem0's own default is 2000, which a reasoning model spends entirely on reasoning
# before emitting any JSON — the extraction then parses as nothing and the system
# stores an empty result while answering HTTP 200. The budget is therefore a pinned
# property of the run, not a number living in a setup script.


def test_a_suite_may_pin_a_chat_max_tokens(root: Path) -> None:
    write_v2(
        root,
        V2_BODY.replace(
            "chat: {id: gpt-4o-mini, provider: openai, temperature: 0, seed: 7}",
            "chat: {id: gpt-4o-mini, provider: openai, temperature: 0, seed: 7, max_tokens: 8000}",
        ),
    )
    assert load_suite("v2", root).models.chat.max_tokens == 8000


def test_a_suite_that_pins_no_max_tokens_still_loads(root: Path) -> None:
    """v1..v3 predate the pin; their rows stay reproducible."""
    write_v2(root, V2_BODY)
    assert load_suite("v2", root).models.chat.max_tokens is None


def test_the_repo_suite_v3_loads_and_verifies() -> None:
    suite = load_suite("v3", Path(__file__).resolve().parents[2])
    assert suite.suite == "v3"
    assert suite.models.chat.max_tokens is None


def test_the_repo_suite_v4_pins_a_chat_max_tokens() -> None:
    """The current suite must pin the budget: an unpinned one silently truncates."""
    suite = load_suite("v4", Path(__file__).resolve().parents[2])
    assert suite.suite == "v4"
    assert suite.models.chat.max_tokens is not None
    assert suite.models.chat.max_tokens >= 4000
