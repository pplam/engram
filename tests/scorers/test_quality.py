"""Quality scorers: mcq (deterministic), judge_binary, exact_match_f1 (§2.10, §7.1)."""

import pytest

from scorers.quality import (
    SCORERS,
    QualityOutcome,
    UnknownScorer,
    get_scorer,
    score_exact_match_f1,
    score_judge_binary,
    score_mcq,
)


def test_registry_exposes_the_three_built_scorers() -> None:
    assert set(SCORERS) == {"mcq", "judge_binary", "exact_match_f1"}


def test_get_scorer_returns_the_named_scorer() -> None:
    assert get_scorer("mcq") is score_mcq


def test_an_unknown_scorer_is_rejected_by_name() -> None:
    with pytest.raises(UnknownScorer, match="telepathy"):
        get_scorer("telepathy")


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ("A", True),
        ("A. Lisbon", True),
        ("Lisbon", True),
        ("The answer is A.", True),
        ("B", False),
        ("Berlin", False),
        ("", False),
    ],
)
def test_mcq_maps_a_prediction_onto_the_gold_option(prediction: str, expected: bool) -> None:
    outcome = score_mcq(
        prediction=prediction,
        gold=("Lisbon",),
        options=("A. Lisbon", "B. Berlin"),
        judge_label=None,
    )
    assert outcome.is_correct is expected


def test_mcq_needs_no_judge() -> None:
    outcome = score_mcq(
        prediction="A", gold=("Lisbon",), options=("A. Lisbon", "B. Berlin"), judge_label=None
    )
    assert outcome.used_judge is False


def test_mcq_without_options_falls_back_to_exact_text() -> None:
    outcome = score_mcq(prediction="Lisbon", gold=("Lisbon",), options=None, judge_label=None)
    assert outcome.is_correct


def test_mcq_is_case_and_whitespace_insensitive() -> None:
    outcome = score_mcq(
        prediction="  lisbon ",
        gold=("Lisbon",),
        options=("A. Lisbon", "B. Berlin"),
        judge_label=None,
    )
    assert outcome.is_correct is True


def test_mcq_does_not_match_a_letter_appearing_inside_a_word() -> None:
    """A bare 'b' inside 'probably' must not select option B."""
    outcome = score_mcq(
        prediction="probably not sure",
        gold=("Lisbon",),
        options=("A. Lisbon", "B. Berlin"),
        judge_label=None,
    )
    assert outcome.is_correct is False


@pytest.mark.parametrize(
    ("label", "expected"),
    [("CORRECT", True), ("WRONG", False), ("UNPARSEABLE", False)],
)
def test_judge_binary_follows_the_judge_label(label: str, expected: bool) -> None:
    outcome = score_judge_binary(
        prediction="anything", gold=("on the mat",), options=None, judge_label=label
    )
    assert outcome.is_correct is expected
    assert outcome.used_judge is True


def test_judge_binary_reports_an_absent_label_as_incorrect_and_flags_it() -> None:
    outcome = score_judge_binary(prediction="x", gold=("y",), options=None, judge_label=None)
    assert outcome.is_correct is False
    assert outcome.detail == "no_judge_label"


def test_exact_match_rewards_an_identical_answer() -> None:
    outcome = score_exact_match_f1(
        prediction="on the mat", gold=("on the mat",), options=None, judge_label=None
    )
    assert outcome.is_correct is True
    assert outcome.exact_match == 1.0
    assert outcome.token_f1 == 1.0


def test_exact_match_normalizes_case_punctuation_and_articles() -> None:
    outcome = score_exact_match_f1(
        prediction="On the Mat.", gold=("the mat",), options=None, judge_label=None
    )
    assert outcome.exact_match == 0.0
    assert outcome.token_f1 is not None
    assert outcome.token_f1 > 0.5


def test_exact_match_takes_the_best_of_several_gold_candidates() -> None:
    outcome = score_exact_match_f1(
        prediction="Ana", gold=("the user's sister Ana", "Ana"), options=None, judge_label=None
    )
    assert outcome.exact_match == 1.0


def test_token_f1_is_partial_for_a_partial_overlap() -> None:
    outcome = score_exact_match_f1(
        prediction="cat on mat", gold=("the cat sat on the mat",), options=None, judge_label=None
    )
    assert outcome.token_f1 is not None
    assert 0.0 < outcome.token_f1 < 1.0


def test_token_f1_is_zero_when_nothing_overlaps() -> None:
    outcome = score_exact_match_f1(
        prediction="completely different", gold=("on the mat",), options=None, judge_label=None
    )
    assert outcome.token_f1 == 0.0


def test_an_empty_prediction_scores_zero_rather_than_raising() -> None:
    outcome = score_exact_match_f1(
        prediction="", gold=("on the mat",), options=None, judge_label=None
    )
    assert outcome.is_correct is False
    assert outcome.token_f1 == 0.0


def test_rouge_l_is_reported_for_exact_match_f1() -> None:
    outcome = score_exact_match_f1(
        prediction="the cat sat", gold=("the cat sat on the mat",), options=None, judge_label=None
    )
    assert outcome.rouge_l is not None
    assert 0.0 < outcome.rouge_l <= 1.0


def test_every_scorer_returns_a_quality_outcome() -> None:
    for scorer in SCORERS.values():
        outcome = scorer(prediction="a", gold=("a",), options=("A. a",), judge_label="CORRECT")
        assert isinstance(outcome, QualityOutcome)


def test_scoring_is_deterministic() -> None:
    prediction, gold = "cat on mat", ("the cat sat on the mat",)
    first = score_exact_match_f1(prediction, gold, options=None, judge_label=None)
    second = score_exact_match_f1(prediction, gold, options=None, judge_label=None)
    assert first == second
