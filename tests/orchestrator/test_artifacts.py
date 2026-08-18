"""Append-only JSONL artifacts; resume is read_ids() then skip (§2.4)."""

import json
from pathlib import Path

import pytest

from orchestrator.artifacts import ArtifactError, JsonlArtifact


@pytest.fixture
def artifact(tmp_path: Path) -> JsonlArtifact:
    return JsonlArtifact(tmp_path / "stage.jsonl")


def test_append_then_stream_round_trips(artifact: JsonlArtifact) -> None:
    artifact.append({"id": "a", "n": 1})
    artifact.append({"id": "b", "n": 2})
    assert list(artifact.stream()) == [{"id": "a", "n": 1}, {"id": "b", "n": 2}]


def test_append_creates_parent_directories(tmp_path: Path) -> None:
    nested = JsonlArtifact(tmp_path / "runs" / "r1" / "stage.jsonl")
    nested.append({"id": "a"})
    assert nested.path.is_file()


def test_one_record_per_line(artifact: JsonlArtifact) -> None:
    artifact.append({"id": "a"})
    artifact.append({"id": "b"})
    assert artifact.path.read_text().splitlines() == ['{"id": "a"}', '{"id": "b"}']


def test_read_ids_on_a_missing_file_is_empty_not_an_error(artifact: JsonlArtifact) -> None:
    assert artifact.read_ids() == set()


def test_stream_on_a_missing_file_is_empty(artifact: JsonlArtifact) -> None:
    assert list(artifact.stream()) == []


def test_read_ids_returns_every_written_id(artifact: JsonlArtifact) -> None:
    for name in ("a", "b", "c"):
        artifact.append({"id": name})
    assert artifact.read_ids() == {"a", "b", "c"}


def test_append_never_rewrites_earlier_records(artifact: JsonlArtifact) -> None:
    artifact.append({"id": "a", "n": 1})
    first_line = artifact.path.read_text().splitlines()[0]
    artifact.append({"id": "a", "n": 99})
    assert artifact.path.read_text().splitlines()[0] == first_line


def test_a_truncated_final_line_is_reported_not_silently_dropped(artifact: JsonlArtifact) -> None:
    artifact.append({"id": "a"})
    with artifact.path.open("a") as handle:
        handle.write('{"id": "b", "n":')
    with pytest.raises(ArtifactError, match="line 2"):
        list(artifact.stream())


def test_read_ids_also_reports_a_truncated_final_line(artifact: JsonlArtifact) -> None:
    artifact.append({"id": "a"})
    with artifact.path.open("a") as handle:
        handle.write('{"id": "b"')
    with pytest.raises(ArtifactError, match="line 2"):
        artifact.read_ids()


def test_a_record_without_an_id_is_rejected_on_append(artifact: JsonlArtifact) -> None:
    with pytest.raises(ArtifactError, match="id"):
        artifact.append({"n": 1})


def test_a_record_without_an_id_is_reported_on_read(artifact: JsonlArtifact) -> None:
    artifact.path.write_text('{"n": 1}\n')
    with pytest.raises(ArtifactError, match="id"):
        artifact.read_ids()


def test_blank_lines_are_tolerated(artifact: JsonlArtifact) -> None:
    artifact.path.write_text('{"id": "a"}\n\n{"id": "b"}\n')
    assert artifact.read_ids() == {"a", "b"}


def test_rerunning_a_stage_over_existing_output_adds_nothing(artifact: JsonlArtifact) -> None:
    planned = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    for record in planned:
        artifact.append(record)
    before = artifact.path.read_text()

    done = artifact.read_ids()
    for record in planned:
        if record["id"] not in done:
            artifact.append(record)
    assert artifact.path.read_text() == before


def test_resume_appends_only_the_missing_records(artifact: JsonlArtifact) -> None:
    artifact.append({"id": "a"})
    done = artifact.read_ids()
    for record in ({"id": "a"}, {"id": "b"}):
        if record["id"] not in done:
            artifact.append(record)
    assert artifact.read_ids() == {"a", "b"}
    assert len(artifact.path.read_text().splitlines()) == 2


def test_append_writes_utf8_unescaped(artifact: JsonlArtifact) -> None:
    artifact.append({"id": "a", "content": "café — naïve"})
    assert "café — naïve" in artifact.path.read_text()


def test_records_are_written_with_stable_key_order(artifact: JsonlArtifact) -> None:
    """Byte-identical output for equal records is what makes plan.jsonl comparable."""
    artifact.append({"id": "a", "b": 1, "a": 2})
    other = JsonlArtifact(artifact.path.with_name("other.jsonl"))
    other.append({"a": 2, "id": "a", "b": 1})
    assert artifact.path.read_text() == other.path.read_text()


def test_count_matches_the_number_of_records(artifact: JsonlArtifact) -> None:
    for name in ("a", "b", "c"):
        artifact.append({"id": name})
    assert artifact.count() == 3


def test_count_on_a_missing_file_is_zero(artifact: JsonlArtifact) -> None:
    assert artifact.count() == 0


def test_stream_does_not_load_the_whole_file(artifact: JsonlArtifact) -> None:
    """stream() is lazy: the first record is available before the file is exhausted."""
    for i in range(100):
        artifact.append({"id": str(i)})
    stream = artifact.stream()
    assert next(iter(stream))["id"] == "0"


def test_write_json_and_read_json_round_trip(tmp_path: Path) -> None:
    from orchestrator.artifacts import read_json, write_json

    path = tmp_path / "manifest.json"
    write_json(path, {"run_id": "r1", "top_k": 100})
    assert read_json(path) == {"run_id": "r1", "top_k": 100}
    assert json.loads(path.read_text())["run_id"] == "r1"


def test_write_json_creates_parent_directories(tmp_path: Path) -> None:
    from orchestrator.artifacts import write_json

    path = tmp_path / "runs" / "r1" / "manifest.json"
    write_json(path, {"a": 1})
    assert path.is_file()


def test_read_json_on_a_missing_file_raises_artifact_error(tmp_path: Path) -> None:
    from orchestrator.artifacts import read_json

    with pytest.raises(ArtifactError, match="absent.json"):
        read_json(tmp_path / "absent.json")
