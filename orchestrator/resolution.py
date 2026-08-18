"""Dataset-then-suite precedence for the judge model and the answer prompt (D7).

Precedence exists here and nowhere else. Both resolved values carry their source
so `manifest.json` and every report row can say which one produced a number.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from orchestrator.datasets import DatasetConfig
from orchestrator.suite import Suite, sha256_text

Source = Literal["suite", "dataset"]


@dataclass(frozen=True)
class ResolvedJudge:
    """The judge that actually scores this dataset, and where it came from.

    `provider` rides along because a pinned model is `(provider, id)` (D11) and the
    dataset's override has to name its own — a judge resolved to a dataset's model but
    the suite's provider would be a pair nobody pinned.
    """

    id: str
    temperature: float
    source: Source
    provider: str | None = None


@dataclass(frozen=True)
class ResolvedPrompt:
    """The prompt text that is actually used, hashed for the manifest."""

    text: str
    sha256: str
    ref: str
    source: Source


def resolve_judge(suite: Suite, dataset: DatasetConfig) -> ResolvedJudge:
    """Return the dataset's `official_judge` if it pins one, else the suite judge."""
    if dataset.official_judge is not None:
        return ResolvedJudge(
            id=dataset.official_judge.id,
            temperature=dataset.official_judge.temperature,
            source="dataset",
            provider=dataset.official_judge.provider,
        )
    return ResolvedJudge(
        id=suite.models.judge.id,
        temperature=suite.models.judge.temperature,
        source="suite",
        provider=suite.models.judge.provider,
    )


def resolve_answer_prompt(suite: Suite, dataset: DatasetConfig) -> ResolvedPrompt:
    """Return the dataset's `official_prompt` if it pins one, else the suite prompt."""
    if dataset.official_prompt is not None:
        path: Path = dataset.root / dataset.official_prompt
        if not path.is_file():
            raise FileNotFoundError(
                f"dataset {dataset.name!r} pins official_prompt "
                f"{dataset.official_prompt}, which is not at {path}"
            )
        text = path.read_text()
        return ResolvedPrompt(
            text=text,
            sha256=sha256_text(text),
            ref=dataset.official_prompt,
            source="dataset",
        )
    return ResolvedPrompt(
        text=suite.prompts.answer.text,
        sha256=suite.prompts.answer.sha256,
        ref=suite.prompts.answer.ref,
        source="suite",
    )
