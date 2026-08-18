"""Task-quality scorers (ARCH §7.1).

Each is a pure function of (prediction, gold, options, judge_label). A dataset
declares which one applies; nothing here branches on dataset name.

`judge_rubric` and `event_ordering` are deliberately absent until a dataset needs
them (§2.10).
"""

import re
import string
from collections.abc import Callable, Sequence
from dataclasses import dataclass

_ARTICLES = {"a", "an", "the"}
_PUNCT = str.maketrans("", "", string.punctuation)


class UnknownScorer(Exception):
    """The dataset declared a scorer that does not exist."""


@dataclass(frozen=True)
class QualityOutcome:
    """One question's quality result, plus the extra measures its scorer produces."""

    is_correct: bool
    used_judge: bool
    exact_match: float | None = None
    token_f1: float | None = None
    rouge_l: float | None = None
    detail: str | None = None


Scorer = Callable[..., QualityOutcome]


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and articles, and collapse whitespace."""
    lowered = text.lower().translate(_PUNCT)
    return " ".join(word for word in lowered.split() if word not in _ARTICLES)


def _tokens(text: str) -> list[str]:
    return normalize(text).split()


def _f1(prediction: str, gold: str) -> float:
    predicted, expected = _tokens(prediction), _tokens(gold)
    if not predicted or not expected:
        return 0.0
    shared = 0
    remaining = list(expected)
    for token in predicted:
        if token in remaining:
            remaining.remove(token)
            shared += 1
    if shared == 0:
        return 0.0
    precision, recall = shared / len(predicted), shared / len(expected)
    return 2 * precision * recall / (precision + recall)


def _lcs(left: Sequence[str], right: Sequence[str]) -> int:
    table = [0] * (len(right) + 1)
    for token in left:
        previous = 0
        for index, other in enumerate(right, start=1):
            current = table[index]
            table[index] = previous + 1 if token == other else max(table[index], table[index - 1])
            previous = current
    return table[-1]


def _rouge_l(prediction: str, gold: str) -> float:
    predicted, expected = _tokens(prediction), _tokens(gold)
    if not predicted or not expected:
        return 0.0
    common = _lcs(predicted, expected)
    if common == 0:
        return 0.0
    precision, recall = common / len(predicted), common / len(expected)
    return 2 * precision * recall / (precision + recall)


def score_mcq(
    prediction: str,
    gold: tuple[str, ...],
    options: tuple[str, ...] | None,
    judge_label: str | None,
) -> QualityOutcome:
    """Map a prediction onto an option deterministically — no judge involved."""
    normalized = normalize(prediction)
    if not options:
        correct = any(normalized == normalize(candidate) for candidate in gold)
        return QualityOutcome(is_correct=correct, used_judge=False, detail="no_options")

    gold_letters = {
        option.split(".", 1)[0].strip().lower()
        for option in options
        if any(normalize(option.split(".", 1)[-1]) == normalize(g) for g in gold)
    }
    # Letters are matched on raw tokens: normalize() strips "a" as an article, which
    # would otherwise erase option A.
    words = set(re.findall(r"[a-z0-9]+", prediction.lower()))
    if gold_letters & words:
        return QualityOutcome(is_correct=True, used_judge=False)

    other_letters = {option.split(".", 1)[0].strip().lower() for option in options} - gold_letters
    if other_letters & words:
        return QualityOutcome(is_correct=False, used_judge=False)

    correct = any(normalize(candidate) in normalized for candidate in gold if candidate)
    return QualityOutcome(is_correct=correct, used_judge=False)


def score_judge_binary(
    prediction: str,
    gold: tuple[str, ...],
    options: tuple[str, ...] | None,
    judge_label: str | None,
) -> QualityOutcome:
    """Follow the judge's CORRECT/WRONG label; an absent label is flagged, not guessed."""
    if judge_label is None:
        return QualityOutcome(is_correct=False, used_judge=True, detail="no_judge_label")
    return QualityOutcome(is_correct=judge_label.upper() == "CORRECT", used_judge=True)


def score_exact_match_f1(
    prediction: str,
    gold: tuple[str, ...],
    options: tuple[str, ...] | None,
    judge_label: str | None,
) -> QualityOutcome:
    """Best-over-candidates exact match, token F1, and ROUGE-L."""
    if not gold:
        return QualityOutcome(is_correct=False, used_judge=False, detail="no_gold")
    exact = max(float(normalize(prediction) == normalize(c)) for c in gold)
    f1 = max(_f1(prediction, c) for c in gold)
    rouge = max(_rouge_l(prediction, c) for c in gold)
    return QualityOutcome(
        is_correct=exact == 1.0,
        used_judge=False,
        exact_match=exact,
        token_f1=f1,
        rouge_l=rouge,
    )


SCORERS: dict[str, Scorer] = {
    "mcq": score_mcq,
    "judge_binary": score_judge_binary,
    "exact_match_f1": score_exact_match_f1,
}


def get_scorer(name: str) -> Scorer:
    """Return the scorer registered under `name`."""
    try:
        return SCORERS[name]
    except KeyError as err:
        raise UnknownScorer(
            f"unknown scorer {name!r}; available: {', '.join(sorted(SCORERS))}"
        ) from err
