"""Tests for the isolation-boundary ID scheme (ARCH §4.4) and run ids (§4.5)."""

import re
from pathlib import Path

import pytest

from orchestrator.ids import (
    SEP,
    IdError,
    ParsedId,
    new_run_id,
    parse,
    question_id,
    request_id,
    resolve_prefix,
    user_id,
)


def test_user_id_has_eval_prefix_and_components() -> None:
    assert user_id("r1", "locomo", "ctx7") == "eval:r1:locomo:ctx7"


def test_request_id_appends_chunk_handle() -> None:
    uid = user_id("r1", "locomo", "ctx7")
    assert request_id(uid, "c7") == "eval:r1:locomo:ctx7:chunk-c7"


def test_question_id_appends_question_handle() -> None:
    uid = user_id("r1", "locomo", "ctx7")
    assert question_id(uid, "q3") == "eval:r1:locomo:ctx7:q-q3"


@pytest.mark.parametrize(
    ("built", "expected"),
    [
        (user_id("r1", "locomo", "ctx7"), ParsedId("r1", "locomo", "ctx7", None, None)),
        (
            request_id(user_id("r1", "locomo", "ctx7"), "c7"),
            ParsedId("r1", "locomo", "ctx7", "c7", None),
        ),
        (
            question_id(user_id("r1", "locomo", "ctx7"), "q3"),
            ParsedId("r1", "locomo", "ctx7", None, "q3"),
        ),
    ],
)
def test_parse_round_trips_every_form(built: str, expected: ParsedId) -> None:
    assert parse(built) == expected


@pytest.mark.parametrize("component", ["r:1", "loco:mo", "ctx:7"])
def test_rejects_component_containing_separator(component: str) -> None:
    with pytest.raises(IdError, match="':'"):
        user_id(component, component, component)


@pytest.mark.parametrize("bad", ["", "  "])
def test_rejects_empty_component(bad: str) -> None:
    with pytest.raises(IdError, match="empty"):
        user_id(bad, "locomo", "ctx7")


def test_different_run_id_yields_different_user_id() -> None:
    """The namespace guarantee: a new run_id is a new namespace."""
    assert user_id("r1", "locomo", "ctx7") != user_id("r2", "locomo", "ctx7")


def test_parse_rejects_unknown_prefix() -> None:
    with pytest.raises(IdError, match="eval:"):
        parse("other:r1:locomo:ctx7")


def test_parse_rejects_unknown_suffix_kind() -> None:
    with pytest.raises(IdError, match="suffix"):
        parse("eval:r1:locomo:ctx7:frob-1")


def test_parse_rejects_too_few_components() -> None:
    with pytest.raises(IdError, match="components"):
        parse("eval:r1:locomo")


def test_a_generated_run_id_is_twelve_hex_chars() -> None:
    """Docker-style: short enough to type, long enough not to collide (§4.5)."""
    assert re.fullmatch(r"[0-9a-f]{12}", new_run_id())


def test_a_generated_run_id_never_contains_the_separator() -> None:
    """A run_id with a ':' would silently break `parse` on every downstream id."""
    assert all(SEP not in new_run_id() for _ in range(100))


def test_generated_run_ids_do_not_collide() -> None:
    ids = {new_run_id() for _ in range(10_000)}
    assert len(ids) == 10_000


def _runs(root: Path, *ids: str) -> Path:
    for run_id in ids:
        (root / run_id).mkdir(parents=True)
    return root


def test_resolve_prefix_returns_the_only_match(tmp_path: Path) -> None:
    _runs(tmp_path, "8c1e04b7f2a9", "3f5a11cc0de2")
    assert resolve_prefix("8c1e", tmp_path) == "8c1e04b7f2a9"


def test_an_exact_match_wins_over_a_longer_id_it_prefixes(tmp_path: Path) -> None:
    """Otherwise naming a run exactly could still be called ambiguous."""
    _runs(tmp_path, "8c1e", "8c1e04b7f2a9")
    assert resolve_prefix("8c1e", tmp_path) == "8c1e"


def test_an_ambiguous_prefix_raises_listing_every_candidate(tmp_path: Path) -> None:
    """Picking the first match is exactly the bug this function exists to prevent."""
    _runs(tmp_path, "8c1e04b7f2a9", "8c1e99887766")
    with pytest.raises(IdError) as err:
        resolve_prefix("8c1e", tmp_path)
    assert "8c1e04b7f2a9" in str(err.value)
    assert "8c1e99887766" in str(err.value)


def test_an_unmatched_prefix_raises_naming_the_prefix(tmp_path: Path) -> None:
    _runs(tmp_path, "8c1e04b7f2a9")
    with pytest.raises(IdError, match="beef"):
        resolve_prefix("beef", tmp_path)


def test_resolve_prefix_ignores_files_beside_the_run_directories(tmp_path: Path) -> None:
    """Only directories are runs; a stray file must not become a candidate."""
    _runs(tmp_path, "8c1e04b7f2a9")
    (tmp_path / "8c1enotes.txt").write_text("scratch")
    assert resolve_prefix("8c1e", tmp_path) == "8c1e04b7f2a9"


def test_resolve_prefix_on_a_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(IdError, match="no runs"):
        resolve_prefix("8c1e", tmp_path / "absent")
