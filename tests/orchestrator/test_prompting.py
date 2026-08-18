"""Prompts are pinned data, not format strings (§2.9)."""

import pytest

from orchestrator.prompting import PromptError, render


def test_substitutes_a_placeholder() -> None:
    assert render("Q: {question}", question="why?") == "Q: why?"


def test_substitutes_several_placeholders() -> None:
    out = render("{a} then {b}", a="first", b="second")
    assert out == "first then second"


def test_leaves_json_braces_untouched() -> None:
    """The committed judge prompt contains a JSON example; str.format would raise."""
    template = 'Answer {question}\nReply: {"label": "CORRECT"}'
    assert render(template, question="why?") == 'Answer why?\nReply: {"label": "CORRECT"}'


def test_a_value_containing_braces_is_inserted_literally() -> None:
    assert render("{question}", question="what is {this}?") == "what is {this}?"


def test_a_missing_placeholder_is_reported() -> None:
    with pytest.raises(PromptError, match="memories"):
        render("Q: {question}", question="why?", memories="none")


def test_repeated_placeholders_are_all_replaced() -> None:
    assert render("{x} and {x}", x="a") == "a and a"
