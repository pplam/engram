"""Suite loading: one immutable YAML file is the pinned world (ARCH §6).

Prompt refs carry a sha256; loading verifies the file against it and aborts on
mismatch. That check is the drift detector the reproducibility claim rests on.

A pinned model is `(provider, id)` from v2 onward (D11): a provider is a base URL plus a
credential in the gateway config, which is what lets a suite pin a model no OpenAI
endpoint serves. `v1` predates that and names bare ids; it stays loadable because every
published row names the suite version it was produced under, and a version's meaning
cannot be edited after the fact. Every later suite must name a provider per model — a
silent `openai` default is how a run ends up measuring a model nobody chose, and on a
multi-provider gateway a bare id restricts nothing.
"""

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

Boundary = Literal["message_or_sentence"]

# Suites written before D11, kept loadable and exempt from the checks below. A closed
# set: nothing is ever added here, because a suite version's meaning is fixed once a row
# has been published under it.
LEGACY: frozenset[str] = frozenset({"v1"})

# What `bench run` pins when nobody says otherwise. It must never be a legacy version: a
# suite that names no provider pins nothing on a multi-provider gateway, and defaulting
# to it would make the D11 bug the default rather than a deliberate choice.
CURRENT_SUITE = "v4"


class SuiteError(Exception):
    """The suite is missing, malformed, or a pinned prompt has drifted."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChatModel(_Frozen):
    """Pinned answer model."""

    id: str
    provider: str | None = None
    temperature: float = 0
    seed: int | None = None
    # The completion budget a system under test must give this model. Pinned rather
    # than left to each system's default because a reasoning model spends the budget
    # on reasoning before it emits any content: too small a cap yields an empty
    # completion, which a memory system parses as "no facts" and stores as nothing
    # while still answering 200. Optional because v1..v3 predate the pin.
    max_tokens: int | None = None


class EmbeddingModel(_Frozen):
    """Pinned embedding model."""

    id: str
    provider: str | None = None
    dimensions: int | None = None


class JudgeModel(_Frozen):
    """Default judge; a dataset's `official_judge` may override it (D7)."""

    id: str
    provider: str | None = None
    temperature: float = 0


class Models(_Frozen):
    """The three pinned model roles."""

    chat: ChatModel
    embedding: EmbeddingModel
    judge: JudgeModel


class Retrieval(_Frozen):
    """Retrieval budget, identical for every system."""

    top_k: int


class Chunking(_Frozen):
    """Harness-side chunking; `max_messages_per_chunk` is the primary bound (D4)."""

    max_messages_per_chunk: int
    max_words_per_chunk: int
    boundary: Boundary


class PromptRef(_Frozen):
    """A content-hashed prompt file, with its verified text."""

    ref: str
    sha256: str
    text: str = ""


class Prompts(_Frozen):
    """Answer and judge prompts."""

    answer: PromptRef
    judge: PromptRef


class StageLimits(_Frozen):
    """Concurrency and retry bounds for one stage."""

    workers: int
    timeout_s: int
    attempts: int


class Limits(_Frozen):
    """Per-stage limits.

    `retry_on` / `no_retry_on` are **vestigial**: retry is now keyed on the exception
    type an adapter raises, not on a status code, so a status list cannot express the
    policy for an adapter that never speaks HTTP. They stay accepted, and ignored, so
    that v1 still loads — every published row names the suite version it was produced
    under, so an old suite has to remain loadable. Every suite after v1 must omit them,
    which `_check_current` enforces.
    """

    ingest: StageLimits
    retrieve: StageLimits
    retry_on: list[int] = []
    no_retry_on: list[int] = []


class Suite(_Frozen):
    """A loaded, verified suite version."""

    suite: str
    models: Models
    retrieval: Retrieval
    chunking: Chunking
    prompts: Prompts
    limits: Limits
    sha256: str = ""


def sha256_text(text: str) -> str:
    """Return the hex sha256 of `text` encoded as UTF-8."""
    return hashlib.sha256(text.encode()).hexdigest()


def _check_current(name: str, suite: Suite) -> None:
    """Raise unless `suite` meets the current shape: providers, and no status lists.

    Enumerating the legacy versions rather than the current ones means a new suite is
    held to today's rules by default. Forgetting to list it would otherwise let a v3
    inherit v1's exemptions, which is the wrong direction to fail in.
    """
    if suite.suite in LEGACY:
        return
    for role in ("chat", "embedding", "judge"):
        if getattr(suite.models, role).provider is None:
            raise SuiteError(
                f"{name}: models.{role} names no provider. A pinned model is "
                "(provider, id) (D11); a provider names a block in gateway/bifrost.json. "
                "Defaulting it would silently measure a model nobody chose."
            )
    for field in ("retry_on", "no_retry_on"):
        if getattr(suite.limits, field):
            raise SuiteError(
                f"{name}: limits.{field} no longer means anything. Retry is keyed on the "
                "exception type an adapter raises, not on an HTTP status, so a status "
                "list cannot express the policy for an adapter that never speaks HTTP."
            )


def _verify_prompt(ref: PromptRef, root: Path) -> PromptRef:
    path = root / ref.ref
    if not path.is_file():
        raise SuiteError(f"prompt file {ref.ref} not found (looked at {path})")
    text = path.read_text()
    actual = sha256_text(text)
    if actual != ref.sha256:
        raise SuiteError(
            f"prompt {ref.ref} has drifted: pinned sha256 {ref.sha256[:12]}..., "
            f"file is {actual[:12]}..."
        )
    return ref.model_copy(update={"text": text})


def load_suite(version: str, root: Path = Path()) -> Suite:
    """Return the verified suite `version` from `root/suites/<version>.yaml`."""
    path = root / "suites" / f"{version}.yaml"
    if not path.is_file():
        raise SuiteError(f"no suite at {path.name} (looked in {root / 'suites'})")

    body = path.read_text()
    raw: Any = yaml.safe_load(body)
    if not isinstance(raw, dict):
        raise SuiteError(f"{path.name} must contain a YAML mapping")

    try:
        suite = Suite.model_validate({**raw, "sha256": sha256_text(body)})
    except ValidationError as err:
        fields = ", ".join(".".join(str(p) for p in e["loc"]) for e in err.errors())
        raise SuiteError(f"{path.name} is invalid ({fields}): {err}") from err

    if suite.suite != version:
        raise SuiteError(f"{path.name} declares suite {suite.suite!r}, expected {version!r}")

    _check_current(path.name, suite)

    return suite.model_copy(
        update={
            "prompts": Prompts(
                answer=_verify_prompt(suite.prompts.answer, root),
                judge=_verify_prompt(suite.prompts.judge, root),
            )
        }
    )
